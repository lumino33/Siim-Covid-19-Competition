import numpy as np
import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from mean_average_precision import MetricBuilder

from models import SiimDetModel
from dataset import SiimDetDataset, classes
from utils import seed_everything, refine_dataframe, collate_fn
from config import Config

import warnings
warnings.filterwarnings("ignore")


SEED = 1234
seed_everything(SEED)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('/home/hungld11/Documents/SIIM COVID DETECTION/dataset/train_kfold.csv')

    for fold in Config.folds:     
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        
        train_dataset = SiimDetDataset(df = train_df, 
                                       image_dirs = Config.image_dirs,
                                       image_size = Config.image_size,
                                       mode = "train")
        val_dataset = SiimDetDataset(df = val_df,               
                                    image_dirs = Config.image_dirs,
                                    image_size = Config.image_size,
                                    mode = "val")
        train_loader = DataLoader(train_dataset, 
                                  batch_size=Config.batch_size,
                                  sampler = RandomSampler(train_dataset),
                                  num_workers = Config.num_workers,
                                  drop_last = True,
                                  collate_fn = collate_fn
                                  )
        val_loader = DataLoader(val_dataset, 
                                  batch_size=Config.batch_size,
                                  sampler = SequentialSampler(val_dataset),
                                  num_workers = Config.num_workers,
                                  drop_last = False,
                                  collate_fn = collate_fn
                                  )
        
        model = SiimDetModel(backbone_name = Config.backbone_name,
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
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.epochs)
        scaler = torch.cuda.amp.GradScaler()

        LOG = '/home/hungld11/Documents/SIIM COVID DETECTION/checkpoints/image_level/{}_{}_aux_fold_{}.txt'.format(Config.backbone_name, Config.image_size, fold)
        CHECKPOINT = '/home/hungld11/Documents/SIIM COVID DETECTION/checkpoints/image_level/{}_{}_aux_fold_{}.pth'.format(Config.backbone_name, Config.image_size, fold)
        
        # if fold == 0:
        #     print("Load weights from:", CHECKPOINT)
        #     model.load_state_dict(torch.load(CHECKPOINT))
        
        val_map_max= 0.0 # max = 0.47718
        best_epoch = 0
        if os.path.isfile(LOG):
            os.remove(LOG)
        log_file = open(LOG, 'a')
        log_file.write('epoch, lr, train_loss, val_map \n')
        log_file.close()

        print('FOLD: {} | TRAIN: {} | VALID: {}'.format(fold, len(train_loader.dataset), len(val_loader.dataset)))
        
        for epoch in range(Config.epochs):
            
            #TRAIN
            model.train()
            train_loss = []

            train_loop = tqdm(train_loader)
            for images, targets in train_loop:
                scheduler.step()
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    det_loss_dict = model(images, targets)
                    loss = sum(l for l in det_loss_dict.values())
                    train_loss.append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch+1, Config.epochs, optimizer.param_groups[0]['lr']))
                train_loop.set_postfix(loss=np.mean(train_loss))
            train_loss = np.mean(train_loss)
            
            #VALIDATION
            model.eval()

            metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

            val_bar_loop = tqdm(val_loader, total=len(val_loader))
            
            for images, targets in val_bar_loop:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast(), torch.no_grad():
                    det_outputs = model(images, targets)

                    for t, d in zip(targets, det_outputs):
                        gt_boxes = t['boxes'].data.cpu().numpy()
                        gt_boxes = np.hstack((gt_boxes, np.zeros((gt_boxes.shape[0], 3), dtype=gt_boxes.dtype)))
                        
                        det_boxes = d['boxes'].data.cpu().numpy()
                        det_scores = d['scores'].data.cpu().numpy()
                        det_scores = det_scores.reshape(det_scores.shape[0], 1)
                        det_pred = np.hstack((det_boxes, np.zeros((det_boxes.shape[0], 1), dtype=det_boxes.dtype), det_scores))
                        metric_fn.add(det_pred, gt_boxes)


            val_map = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1), mpolicy='soft')['mAP']

            print('train loss: {:.5f} | val_map: {:.5f}'.format(train_loss, val_map))
            log_file = open(LOG, 'a')
            log_file.write('{}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                epoch, optimizer.param_groups[0]['lr'], train_loss, val_map))
            log_file.close()

            if val_map > val_map_max:
                print('Valid map improved from {:.5f} to {:.5f} saving model to {}'.format(val_map_max, val_map, CHECKPOINT))
                val_map_max = val_map
                best_epoch = epoch
                torch.save(model.state_dict(), CHECKPOINT)

        log_file = open(LOG, 'a')
        log_file.write('Best epoch {} | mAP max: {}\n'.format(best_epoch, val_map_max))
        log_file.close()
        print('Best epoch {} | mAP max: {}'.format(best_epoch, val_map_max))
        
if __name__ == "__main__":
    main()