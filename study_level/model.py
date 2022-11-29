import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.base import SegmentationHead


class SiimClsModel(nn.Module):
    def __init__(self,
                 encoder_name = "efficientnet-b5",
                 encoder_weights = None, # ['imagenet', 'advprop'] or None
                
                 decoder_channels = [256, 128, 64, 32, 16],
                 
                 num_classes = 4,
                 in_features = 2048 #the number of nodes
                 ):
        
        super(SiimClsModel, self).__init__()
        #encoder
        self.encoder = get_encoder(name=encoder_name,
                                   in_channels=3,
                                   depth=5,
                                   weights=encoder_weights,)
        
        #decoder
        self.decoder = UnetDecoder(encoder_channels = self.encoder.out_channels,
                                   decoder_channels = decoder_channels,
                                   n_blocks = 5,
                                   use_batchnorm = True,
                                   )
        # self.decoder = UnetPlusPlusDecoder(encoder_channels = self.encoder.out_channels,
        #                            decoder_channels = decoder_channels,
        #                            n_blocks = 5,
        #                            use_batchnorm = True,
        #                            )
        #classification head
        # self.in_features = in_features
        # self.hidded_layers = nn.Sequential(*list(self.encoder.children())[-5:])
        # self.fc1 = nn.Linear(self.in_features, 1024, bias=True)
        self.in_features = in_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # self.cls_head = nn.Linear(1024, num_classes, bias=True)   
        self.cls_head = nn.Linear(self.in_features, num_classes, bias=True)
        
        #segmentation head
        self.seg_head = SegmentationHead(in_channels=decoder_channels[-1],
                                        out_channels=1,
                                        activation="sigmoid",
                                        )
    
    #@autocast()
    def forward(self, x):
        encoder_features = self.encoder(x)

        #classification
        # hidden_feature = self.hidded_layers(encoder_features[-1])
        # y_cls = self.fc1(hidden_feature.view(-1, self.in_features))
        # y_cls = F.relu(y_cls)
        # y_cls = F.dropout(y_cls, p=0.5)
        # y_cls = self.cls_head(y_cls)
        
        # hidden_feature = self.hidded_layers(encoder_features[-1])
        # y_cls = F.relu(hidden_feature.view(-1, self.in_features))
        # y_cls = F.dropout(y_cls, p=0.5)
        # y_cls = self.cls_head(y_cls)
        
        y_cls = self.global_pool(encoder_features[-1])[:, :, 0, 0]
        y_cls = self.relu(y_cls)
        y_cls = self.dropout(y_cls)
        y_cls = self.cls_head(y_cls)
        
        #segmentation
        y_seg = self.decoder(*encoder_features)
        y_seg = self.seg_head(y_seg)
        
        return y_cls, y_seg 