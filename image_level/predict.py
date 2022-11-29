import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

from utils import seed_everything, collate_fn, refine_det
from config import Config
from models import SiimDetModel
from dataset import SiimDetTestDataset, classes

import warnings
warnings.filterwarnings("ignore")

SEED = 1234
seed_everything(SEED)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv("/home/hungld11/Documents/SIIM COVID DETECTION/dataset/test_meta.csv")
    
    models = {}
    for fold in Config.folds:
        CHECKPOINT = '/home/hungld11/Documents/SIIM COVID DETECTION/checkpoints/image_level/{}_{}_aux_fold_{}.pth'.format(Config.backbone_name, Config.image_size, str(fold))
        print("Loading checkpoint from:", CHECKPOINT)
        models[fold] = SiimDetModel(backbone_name = Config.backbone_name,
                            imagenet_pretrained=True,
                            num_classes= len(classes),
                            in_features=2048, 
                            backbone_pretrained_path=None, 
                            backbone_pretrained_cls_num_classes=None,
                            model_pretrained_path=None, 
                            model_pretrained_cls_num_classes=None, 
                            trainable_layers=Config.trainable_layers,
                            returned_layers=None,
                            extra_blocks=None,)
        models[fold].to(device)
        models[fold].load_state_dict(torch.load(CHECKPOINT))
        models[fold].eval()
        
    test_dataset = SiimDetTestDataset(df=test_df, 
                                      images_dir= Config.test_image_dirs, 
                                      image_size= Config.image_size)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size= Config.batch_size, shuffle=False, 
                             num_workers= Config.num_workers, pin_memory=False, 
                             drop_last=False, collate_fn=collate_fn)

    predict_dict = {}
    for ids, images, targets, heights, widths in tqdm(test_loader, total=len(test_loader)):
        hflip_images = torch.stack(images)
        vflip_images = torch.stack(images)
        hflip_images = torch.flip(hflip_images, dims=(3,)).contiguous()
        vflip_images = torch.flip(vflip_images, dims=(2,)).contiguous()
        
        images = list(image.cuda() for image in images)
        hflip_images = list(image.cuda() for image in hflip_images)
        vflip_images = list(image.cuda() for image in vflip_images)
        
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(), torch.no_grad():
            for fold in Config.folds:
                det_outputs = models[fold](images, targets)
                hflip_det_outputs = models[fold](hflip_images, targets)
                vflip_det_outputs = models[fold](vflip_images, targets)
                for imageid, det, hflip_det, vflip_det, height, width in zip(ids, det_outputs, hflip_det_outputs, vflip_det_outputs, heights, widths):
                    if imageid not in list(predict_dict.keys()):
                        predict_dict[imageid] = [[],[],[], width, height]
                        
                    box_pred = det['boxes'].data.cpu().numpy().astype(float)
                    score_pred = det['scores'].data.cpu().numpy()
                    box_pred = box_pred/float(Config.image_size)
                    box_pred = box_pred.clip(0,1)
                    label_pred = np.zeros_like(score_pred, dtype=int)
                    box_pred, label_pred, score_pred = refine_det(box_pred, label_pred, score_pred)

                    hflip_box_pred = hflip_det['boxes'].data.cpu().numpy().astype(float)
                    hflip_box_pred = hflip_box_pred/float(Config.image_size)
                    hflip_box_pred = hflip_box_pred.clip(0,1)
                    hflip_box_pred[:,[0,2]] = 1 - hflip_box_pred[:,[0,2]]
                    hflip_score_pred = hflip_det['scores'].data.cpu().numpy()
                    hflip_label_pred = np.zeros_like(hflip_score_pred, dtype=int)
                    hflip_box_pred, hflip_label_pred, hflip_score_pred = refine_det(hflip_box_pred, hflip_label_pred, hflip_score_pred)

                    vflip_box_pred = vflip_det['boxes'].data.cpu().numpy().astype(float)
                    vflip_box_pred = vflip_box_pred/float(Config.image_size)
                    vflip_box_pred = vflip_box_pred.clip(0,1)
                    vflip_box_pred[:,[1,3]] = 1 - vflip_box_pred[:,[1,3]]
                    vflip_score_pred = vflip_det['scores'].data.cpu().numpy()
                    vflip_label_pred = np.zeros_like(vflip_score_pred, dtype=int)
                    vflip_box_pred, vflip_label_pred, vflip_score_pred = refine_det(vflip_box_pred, vflip_label_pred, vflip_score_pred)

                    predict_dict[imageid][0] += [box_pred, hflip_box_pred, vflip_box_pred]
                    predict_dict[imageid][1] += [score_pred, hflip_score_pred, vflip_score_pred]
                    predict_dict[imageid][2] += [label_pred, hflip_label_pred, vflip_label_pred]

    
    pred_dict_path = '/home/hungld11/Documents/SIIM COVID DETECTION/results/image_level/{}_{}_fold_{}_test_pred_1.pth'.format(Config.backbone_name, Config.image_size, '_'.join(str(x) for x in Config.folds))
    torch.save(predict_dict, pred_dict_path)
    
if __name__ == "__main__":
    main()