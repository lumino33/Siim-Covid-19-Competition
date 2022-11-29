import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = [
    'Negative for Pneumonia',
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance'
]

class SiimDetDataset(Dataset):
    def __init__(self, df, image_dirs, image_size, mode="test"):
        super(SiimDetDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.image_size = image_size
        self.mode = mode
        assert mode in ["train", "val", "test"]
        
        if self.mode == "train":
            self.transform = albu.Compose([
                # albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.Resize(self.image_size, self.image_size),
                albu.Rotate(limit=10, interpolation=1, border_mode=0, value=0, p=0.25),
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
                ], p=0.5),
                # albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        
    def __len__(self):
        return(len(self.df))
        
    def __getitem__(self, index):
        image_path = '{}/{}.png'.format(self.image_dirs, self.df['imageid'].iloc[index])
        image = Image.open(image_path).convert("L")
        image = np.asarray(image)
        image = np.stack([image, image, image], axis=-1)

        height, width = image.shape[0:2]
        boxes = []
        
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
                boxes.append([x1, y1, x2, y2])  
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=float)
            catids = np.ones(boxes.shape[0], dtype=int)
        else:
            boxes = np.array([], dtype=float).reshape(0,4)
            catids = np.array([], dtype=int)
        
        transformation = self.transform(image=image, bboxes=boxes, category_ids=catids)
        image = transformation["image"]
        boxes = transformation['bboxes']
        boxes = np.array(boxes, dtype=float)
        if boxes.shape[0] > 0:
            target = {}
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)
            target['area'] = torch.as_tensor(area, dtype=torch.float32)
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return image, target
            

class SiimDetTestDataset(Dataset):
    def __init__(self, df, images_dir, image_size):
        super(SiimDetTestDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size

        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = '{}/{}.png'.format(self.images_dir, self.df['imageid'].iloc[index])
        image = Image.open(image_path).convert("L")
        image = np.asarray(image)
        image = np.stack([image, image, image], axis=-1)
        
        height, width = image.shape[0:2]

        image = self.transform(image=image)['image']
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
            "area": torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }
        
        return self.df['imageid'].iloc[index], image, target, height, width