import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = [
    'Negative for Pneumonia',
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance'
]

class SiimClsDataset(Dataset):
    def __init__(self, df, image_dirs, image_size, mode="test"):
        self.df = df
        self.image_dirs = image_dirs
        self.image_size = image_size
        assert mode in ["train", "test", "val"]
        self.mode = mode
        if self.mode == "train":
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0, value=0, p=0.25),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.25),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.RandomBrightnessContrast(),            
                ], p=0.25),
                albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])
    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, index):
        image_path = '{}/{}.png'.format(self.image_dirs, self.df['imageid'].iloc[index])
        image = Image.open(image_path).convert("L")
        image = np.asarray(image)
        image = np.stack([image, image, image], axis=-1)

        label = torch.FloatTensor(self.df.loc[index, classes])
        
        #create mask 
        height, width = image.shape[0:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.df.loc[index, 'hasbox']:
            arr = self.df.loc[index, 'label'].split(' ')
            nums = len(arr) // 6
            assert nums > 0

            for i in range(nums):
                class_name = arr[6*i]
                assert class_name == 'opacity'
                x1 = int(float(arr[6*i+2]))
                y1 = int(float(arr[6*i+3]))
                x2 = int(float(arr[6*i+4]))
                y2= int(float(arr[6*i+5]))
                
                x1 = min(max(0,x1),width)
                x2 = min(max(0,x2),width)
                y1 = min(max(0,y1),height)
                y2 = min(max(0,y2),height)

                if x1 >= x2 or y1 >= y2:
                    continue
                mask[y1:y2,x1:x2] = np.ones((y2-y1, x2-x1), dtype=np.uint8)
        transformation = self.transform(image=image, mask=mask)
        image, mask = transformation["image"], transformation["mask"]
        
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)
        
        if self.mode == "train":
            return image, label, mask
        else:
            return image, label, mask, self.df['imageid'].iloc[index]



class SiimClsTestDataset(Dataset):
    def __init__(self, df, image_dirs, image_size):
        super(SiimClsTestDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.image_size = image_size

        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = '{}/{}.png'.format(self.image_dirs, self.df['imageid'].iloc[index])
        image = Image.open(image_path).convert("L")
        image = np.asarray(image)
        image = np.stack([image, image, image], axis=-1)

        #center crop
        height, width = image.shape[0:2]
        new_size = int(0.8*min(height, width))
        x1 = (width - new_size)//2
        y1 = (height - new_size)//2
        image_center_crop = image[y1:y1+new_size, x1:x1+new_size, :]

        image = self.transform(image=image)['image']
        image_center_crop = self.transform(image=image_center_crop)['image']

            
        return self.df.loc[index, 'imageid'], image, image_center_crop