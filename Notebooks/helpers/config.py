import torch
import os

# --- HYPERPARAMETERS ---
TARGET_CLASS = 'MEL'
IMAGE_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 300
TIMESTEPS = 300  # T value for Diffusion

# --- HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PATHS ---
SPLIT_PATH = '/content/drive/MyDrive/FYP/splits/train_split.csv'
ZIP_PATH = '/content/drive/MyDrive/FYP/archive.zip'
EXTRACT_PATH = '/content/dataset'
# Auto-detect or default path
IMAGE_DIR = '/content/dataset/ISIC2018_Task3_Training_Input'

# Outputs
MODEL_SAVE_DIR = f'/content/drive/MyDrive/FYP/models/diffusion_{TARGET_CLASS}'
SYNTHETIC_SAVE_DIR = f'/content/drive/MyDrive/FYP/synthetic_dataset/{TARGET_CLASS}'