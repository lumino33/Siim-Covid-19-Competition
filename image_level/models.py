from collections import OrderedDict
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch.cpu.amp import autocast
import timm

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class MyBackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(MyBackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
    

class MyGeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Arguments:
        backbone (nn.Module):
        rpn (nn.Module) (region proposal network):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(MyGeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)
        
class MyFasterRCNN(MyGeneralizedRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=512, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(MyFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

class PretrainModel(nn.Module):
    def __init__(self,
        backbone_name='resnet101d',
        imagenet_pretrained=False,
        num_classes=4,
        in_features=2048, 
        backbone_pretrained_path=None, 
        backbone_pretrained_num_classes=None,
    ):
        super(PretrainModel, self).__init__()
        self.in_features = in_features
        if backbone_pretrained_path is None:
            if backbone_name == 'resnet200d' or backbone_name == 'seresnet152d' or backbone_name == 'resnet101d':
                self.backbone = timm.create_model(backbone_name, pretrained=imagenet_pretrained, num_classes=0)
                self.backbone.inplanes = 2048
        else:
            print('Load pretrain: {}'.format(backbone_pretrained_path))
            model = PretrainModel(
                backbone_name=backbone_name,
                imagenet_pretrained=imagenet_pretrained,
                num_classes=backbone_pretrained_num_classes,
                in_features=in_features, 
                backbone_pretrained_path=None, 
                backbone_pretrained_num_classes=None)
            model.load_state_dict(torch.load(backbone_pretrained_path))
            self.backbone = model.backbone
            del model

        self.fc = nn.Linear(in_features, 1024, bias=True)
        self.cls_head = nn.Linear(1024, num_classes, bias=True)

        initialize_head(self.fc)
        initialize_head(self.cls_head)

    @autocast()
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.in_features)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.cls_head(x)
        return x

class SiimDetModel(nn.Module):
    def __init__(self,
        backbone_name='resnet101d',
        imagenet_pretrained=True,
        num_classes=6,
        in_features=2048, 
        backbone_pretrained_path=None, 
        backbone_pretrained_cls_num_classes=None,
        model_pretrained_path=None, 
        model_pretrained_cls_num_classes=None, 
        trainable_layers=5,
        returned_layers=None,
        extra_blocks=None,
        **kwargs
    ):
        super(SiimDetModel, self).__init__()
        self.in_features = in_features

        if model_pretrained_path is None:
            if backbone_pretrained_path is None:
                model = PretrainModel(
                    backbone_name=backbone_name,
                    imagenet_pretrained=imagenet_pretrained,
                    num_classes=num_classes,
                    in_features=in_features, 
                    backbone_pretrained_path=None, 
                    backbone_pretrained_num_classes=None
                )
            else:
                print('Load pretrain: {}'.format(backbone_pretrained_path))
                model = PretrainModel(
                    backbone_name=backbone_name,
                    imagenet_pretrained=imagenet_pretrained,
                    num_classes=backbone_pretrained_cls_num_classes,
                    in_features=in_features, 
                    backbone_pretrained_path=None, 
                    backbone_pretrained_num_classes=None
                )
                model.load_state_dict(torch.load(backbone_pretrained_path))
            backbone = model.backbone
            # del model
            
            assert trainable_layers <= 5 and trainable_layers >= 0
            layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
            # freeze layers only if pretrained backbone is used
            for name, parameter in backbone.named_parameters():
                if all([not name.startswith(layer) for layer in layers_to_train]):
                    parameter.requires_grad_(False)

            if extra_blocks is None:
                extra_blocks = LastLevelMaxPool()

            if returned_layers is None:
                returned_layers = [1, 2, 3, 4]
            assert min(returned_layers) > 0 and max(returned_layers) < 5
            return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

            in_channels_stage2 = backbone.inplanes // 8
            in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
            out_channels = 256
            self.model = MyFasterRCNN(MyBackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks), 2)

        else:
            print('Load pretrain: {}'.format(model_pretrained_path))
            model = SiimDetModel(
                backbone_name=backbone_name,
                imagenet_pretrained=False,
                num_classes=model_pretrained_cls_num_classes,
                in_features=in_features, 
                backbone_pretrained_path=None, 
                backbone_pretrained_num_classes=None,
                model_pretrained_path=None, 
                model_pretrained_cls_num_classes=None, 
            )
            model.load_state_dict(torch.load(model_pretrained_path))
            self.model = model.model
            del model

    @autocast()
    def forward(self, x, t):
        det = self.model(x, t)
        return det