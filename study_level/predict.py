import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import seed_everything, get_study_map
from config import Config
from model import SiimClsModel
from dataset import SiimClsTestDataset

SEED = 1234
seed_everything(SEED)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv("/home/hungld11/Documents/SIIM COVID DETECTION/dataset/test_meta.csv")
    
    models = {}
    for fold in Config.folds:
        CHECKPOINT = '/home/hungld11/Documents/SIIM COVID DETECTION/checkpoints/study_level/{}_{}_UnetDecoder_aux_fold_{}.pth'.format(Config.encoder_name, Config.image_size, fold)
        print("Loading checkpoint from:", CHECKPOINT)
        models[fold] = SiimClsModel(encoder_name = Config.encoder_name,
                            encoder_weights = Config.encoder_weights, 
                            decoder_channels = Config.decoder_channels,
                            num_classes = Config.num_classes,
                            in_features = Config.in_features)
        models[fold].load_state_dict(torch.load(CHECKPOINT))
        models[fold].to(device)
        models[fold].eval()
    
     
    test_dataset = SiimClsTestDataset(df=test_df, 
                                      image_dirs='/home/hungld11/Documents/SIIM COVID DETECTION/dataset/test', 
                                      image_size=512)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=16,
                             shuffle=False,
                             num_workers=1)
    
    preds = []
    imageids = []
    
    for ids, images, images_center_crop in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        images_center_crop = images_center_crop.to(device)
        imageids.extend(ids)
        
        pred = []
        with torch.cuda.amp.autocast(), torch.no_grad():
            for fold in Config.folds:
                pred.append(torch.sigmoid(models[fold](images)[0]))
                pred.append(torch.sigmoid(models[fold](torch.flip(images, dims=(3,)).contiguous())[0]))
                pred.append(torch.sigmoid(models[fold](torch.flip(images, dims=(2,)).contiguous())[0]))
                pred.append(torch.sigmoid(models[fold](torch.flip(images, dims=(2,3)).contiguous())[0]))
                pred.append(torch.sigmoid(models[fold](images_center_crop)[0]))
                pred.append(torch.sigmoid(models[fold](torch.flip(images_center_crop, dims=(3,)).contiguous())[0]))
                pred.append(torch.sigmoid(models[fold](torch.flip(images_center_crop, dims=(2,)).contiguous())[0]))
                pred.append(torch.sigmoid(models[fold](torch.flip(images_center_crop, dims=(2,3)).contiguous())[0]))
                            
        pred = torch.mean(torch.stack(pred, -1),-1).data.cpu().numpy()
        preds.append(pred)
        del pred
    
    preds = np.concatenate(preds, axis=0)
    imageids = np.array(imageids)
    
    pred_dict = dict(zip(imageids, preds))
    pred_dict_path = '/home/hungld11/Documents/SIIM COVID DETECTION/results/study_level/{}_{}_UnetDecoder_aux_fold{}_test_pred_8tta.pth'.format(Config.encoder_name, Config.image_size, '_'.join(str(x) for x in Config.folds))
    torch.save({'pred_dict': pred_dict,}, pred_dict_path)
        
if __name__ == "__main__":
    main()