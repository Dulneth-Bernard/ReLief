import collections
import csv
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# - Implement LesionDataset
#        st implement the __init__, __len__ and __getitem__ methods.
#
#        The __init__ function should have the following prototype
#          def __init__(self, img_dir, labels_fname):
#            - img_dir is the directory path with all the image files
#            - labels_fname is the csv file with image ids and their 
#              corresponding labels
#
#        Note: You should not open all the image files in your __init__.
#              Instead, just read in all the file names into a list and
#              open the required image file in the __getitem__ function.
#              This prevents the machine from running out of memory.
#
# Add augment flag to LesionDataset, so the __init__ function
#                now look like this:
#                   def __init__(self, img_dir, labels_fname, augment=False):
#

class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, labels_fname, augment=False, five_crop=False):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_fname)
        self.image_names = self.labels_df['image'].tolist()
        self.labels = self.labels_df[[col for col in self.labels_df.columns if col not in ['image']]].values
        self.augment = augment
        self.five_crop = five_crop

        if five_crop:
            # FiveCrop returns a tuple of 5 images, so we need to stack them as tensors
            self.transform = transforms.Compose([
                transforms.FiveCrop((200, 300)),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
            ])
        else:
            base_transforms = [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ]
            if augment:
                aug_transforms = [
                    # transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ]
                self.transform = transforms.Compose(aug_transforms + base_transforms)
            else:
                self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        # Avoid errors for images without jpg extnsions in input, label = ds[0]
        if not img_name.lower().endswith('.jpg'):
            img_name = img_name + '.jpg'
        img_path = str(Path(self.img_dir) / img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
