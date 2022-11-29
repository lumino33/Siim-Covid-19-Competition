import pandas as pd
import os 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from segmentation_models_pytorch.utils.losses import DiceLoss
from transformers import get_linear_schedule_with_warmup
from segmentation_models_pytorch.utils.metrics import IoU
from tqdm import tqdm

from utils import seed_everything, get_study_map
from config import Config
from model import SiimClsModel
from dataset import SiimClsDataset

torch.backends.cudnn.benchmark = True

SEED = 1234
seed_everything(SEED)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('/home/hungld11/Documents/SIIM COVID DETECTION/dataset/train_kfold.csv')
    
    for fold in Config.folds:     
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        
        train_dataset = SiimClsDataset(df = train_df, 
                                       image_dirs = Config.image_dirs,
                                       image_size = Config.image_size,
                                       mode = "train")
        val_dataset = SiimClsDataset(df = val_df,               
                                    image_dirs = Config.image_dirs,
                                    image_size = Config.image_size,
                                    mode = "val")
        train_loader = DataLoader(train_dataset, 
                                  batch_size=Config.batch_size,
                                  sampler = RandomSampler(train_dataset),
                                  num_workers = Config.num_workers,
                                  drop_last = True
                                  )
        val_loader = DataLoader(val_dataset, 
                                  batch_size=Config.batch_size,
                                  sampler = SequentialSampler(val_dataset),
                                  num_workers = Config.num_workers,
                                  drop_last = False
                                  )
        
        model = SiimClsModel(encoder_name = Config.encoder_name,
                            encoder_weights = Config.encoder_weights, 
                            decoder_channels = Config.decoder_channels,
                            num_classes = Config.num_classes,
                            in_features = Config.in_features)
        model.to(device)
        
        cls_criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([0.2, 0.2, 0.3, 0.3])).to(device)
        seg_criterion = DiceLoss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
        
        num_warmup_steps = 0 
        num_train_steps = int(Config.epochs*len(train_loader)) 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        
        scaler = torch.cuda.amp.GradScaler()
        
        LOG = '/home/hungld11/Documents/SIIM COVID DETECTION/checkpoints/study_level/{}_{}_{}_aux_fold_{}.txt'.format(Config.encoder_name, Config.image_size, "Unet++Decoder", fold)
        CHECKPOINT = '/home/hungld11/Documents/SIIM COVID DETECTION/checkpoints/study_level/{}_{}_{}_aux_fold_{}.pth'.format(Config.encoder_name, Config.image_size, "Unet++Decoder", fold)
        val_map_max= 0.33220

        if os.path.isfile(LOG):
            os.remove(LOG)
        log_file = open(LOG, 'a')
        log_file.write('epoch, lr, train_loss, train_iou, val_map \n')
        log_file.close()

        best_epoch = 0

        iou_func = IoU(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None)
        
        print('FOLD: {} | TRAIN: {} | VALID: {}'.format(fold, len(train_loader.dataset), len(val_loader.dataset)))
        
        for epoch in range(Config.epochs):
            #training
            model.train()
            train_loss = []
            train_iou = []
            bar_loop = tqdm(train_loader, total=len(train_loader))
            
            for image, label, mask in bar_loop:
                image, label, mask = image.to(device), label.to(device), mask.to(device)
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    cls_outputs, seg_outputs = model(image)
                    cls_loss = cls_criterion(cls_outputs, label)
                    seg_loss = seg_criterion(seg_outputs, mask)
                    loss = 0.6*cls_loss + 0.4*seg_loss
                    iou_score = iou_func(seg_outputs, mask)
                        
                    train_iou.append(iou_score.item())
                    train_loss.append(loss.item())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                bar_loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch+1, Config.epochs, optimizer.param_groups[0]['lr']))
                bar_loop.set_postfix(loss=np.mean(train_loss), iou=np.mean(train_iou))
                
            train_loss = np.mean(train_loss)
            train_iou = np.mean(train_iou)
            
            #validation
            model.eval()
            cls_preds = []
            imageids = []
            val_bar_loop = tqdm(val_loader, total=len(val_loader))
            
            for image, label, mask, imageid in val_bar_loop:
                image = image.to(device) 

                with torch.cuda.amp.autocast(), torch.no_grad():
                    val_cls_outputs, _ = model(image)
                    cls_preds.append(torch.sigmoid(val_cls_outputs).data.cpu().numpy())
                    imageids.extend(imageid)
                    
            cls_preds = np.vstack(cls_preds)
            imageids = np.array(imageids)
            
            pred_dict = dict(zip(imageids, cls_preds))
            
            val_map = get_study_map(val_df, pred_dict, stride=0.001)['mAP']
            
            print('train loss: {:.5f} | train iou: {:.5f} | val_map: {:.5f}' \
                  .format(train_loss, train_iou, val_map))
            
            log_file = open(LOG, 'a')
            log_file.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                epoch, optimizer.param_groups[0]['lr'], train_loss, train_iou, val_map))
            log_file.close()
            
            if val_map > val_map_max:
                print('Ema valid map improved from {:.5f} to {:.5f} saving model to {}'.format(val_map_max, val_map, CHECKPOINT))
                val_map_max = val_map
                best_epoch = epoch
                torch.save(model.state_dict(), CHECKPOINT)
    
        log_file = open(LOG, 'a')
        log_file.write('Best epoch {} | mAP max: {}\n'.format(best_epoch, val_map_max))
        log_file.close()
        print('Best epoch {} | mAP max: {}'.format(best_epoch, val_map_max))
        
if __name__ == "__main__":
    main()