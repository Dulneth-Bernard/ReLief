import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import zipfile
from torchvision import transforms
import config

def prepare_data():
    """ Auto-extracts zip if needed """
    if not os.path.exists(config.IMAGE_DIR):
        print("📦 Unzipping dataset...")
        with zipfile.ZipFile(config.ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(config.EXTRACT_PATH)
            
        # Update config.IMAGE_DIR dynamically if needed
        for root, dirs, files in os.walk(config.EXTRACT_PATH):
            if len([f for f in files if f.endswith('.jpg')]) > 5:
                print(f"📸 Images located in: {root}")
                return root
    return config.IMAGE_DIR

class TargetedDataset(Dataset):
    def __init__(self, csv_file, img_dir, target_label, transform=None):
        df = pd.read_csv(csv_file)
        self.data = df[df['label'] == target_label].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx]['image'])
        if not img_name.endswith('.jpg'): img_name += '.jpg'
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image

def get_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])