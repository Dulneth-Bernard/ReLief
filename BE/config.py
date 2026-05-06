# Backend Configuration
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_weights", "checkpoints")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Model Configuration
CLASS_NAMES = ['NV', 'MEL', 'BKL', 'BCC', 'AKIEC', 'VASC', 'DF']
CLASS_FULL_NAMES = {
    'NV': 'Melanocytic Nevus',
    'MEL': 'Melanoma',
    'BKL': 'Benign Keratosis',
    'BCC': 'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratosis',
    'VASC': 'Vascular Lesion',
    'DF': 'Dermatofibroma'
}

# Available Classification Models
CLASSIFICATION_MODELS = {
    "run4_hybrid_no_cost_aware": {
        "id": "run4_hybrid_no_cost_aware",
        "name": "EfficientNet-B0 — Hybrid Dataset (No Cost-Aware)",
        "description": "Trained on a hybrid dataset consisting of both real and diffusion-generated synthetic images. It used standard Cross-Entropy loss without cost-aware class weighting and without dynamic on-the-fly augmentations.",
        "weights_file": "run4_hybrid_no_cost_aware.pth",
        "architecture": "EfficientNet-B0",
        "accuracy": "76.01% Balanced Accuracy / 77.23% Macro F1",
        "type": "Baseline (Hybrid Dataset)"
    },
    "run5_hybrid_cost_aware": {
        "id": "run5_hybrid_cost_aware",
        "name": "EfficientNet-B0 — Hybrid Dataset (Cost-Aware)",
        "description": "Trained on the hybrid dataset integrating standard Inverse Frequency cost-aware class weighting to penalise minority class misclassifications, but without on-the-fly augmentations.",
        "weights_file": "run5_hybrid_cost_aware.pth",
        "architecture": "EfficientNet-B0",
        "accuracy": "79.17% Balanced Accuracy / 79.43% Macro F1",
        "type": "Cost-Aware Baseline"
    },
    "run6a_balanced_minority_cost_aware": {
        "id": "run6a_balanced_minority_cost_aware",
        "name": "EfficientNet-B0 — Balanced Minority Oversampling (Cost-Aware)",
        "description": "Trained using a targeted dataset where severe minority classes were augmented to balance the distribution against the real-world data, combined with cost-aware learning to improve minority recall.",
        "weights_file": "run6a_balanced_minority_cost_aware.pth",
        "architecture": "EfficientNet-B0",
        "accuracy": "~78.00% Macro F1",
        "type": "Experimental (Minority Oversampled)"
    },
    "run6b_balanced_capped_nv_cost_aware": {
        "id": "run6b_balanced_capped_nv_cost_aware",
        "name": "EfficientNet-B0 — Balanced Capped-NV (Cost-Aware)",
        "description": "Trained on a strictly constrained dataset where the majority class (NV) was artificially capped to prevent mathematical bias. Utilised on-the-fly augmentations and standard inverse frequency cost-aware weights.",
        "weights_file": "run6b_balanced_capped_nv_cost_aware.pth",
        "architecture": "EfficientNet-B0",
        "accuracy": "85.22% Validation Accuracy / 78.03% Macro F1",
        "type": "Experimental (Capped Majority)"
    },
    "e5_dataset_a_moderate_balanced_hybrid": {
        "id": "e5_dataset_a_moderate_balanced_hybrid",
        "name": "EfficientNet-B0 — Two-Stage Transfer Learning",
        "description": "Trained on a moderately balanced hybrid dataset using a formal two-stage transfer learning approach (training a frozen head first, followed by full-unfreeze fine-tuning) with dynamic augmentations and cost-aware weighting.",
        "weights_file": "e5_dataset_a_moderate_balanced_hybrid.pth",
        "architecture": "EfficientNet-B0",
        "accuracy": "80.23% Validation Accuracy / 78.70% Macro F1",
        "type": "Two-Stage Transfer Learning"
    },
    "run14_mobilenetv2_ldam": {
        "id": "run14_mobilenetv2_ldam",
        "name": "MobileNetV2 — LDAM Cost-Aware (Run 14)",
        "description": "Trained on a strictly balanced hybrid dataset using dynamic on-the-fly augmentations and the advanced Label-Distribution-Aware Margin (LDAM) loss to aggressively enforce safety margins for high-risk minority classes.",
        "weights_file": "run14_mobilenetv2_ldam.pth",
        "architecture": "MobileNetV2",
        "accuracy": "90.69% Test Accuracy / 91.35% Balanced Accuracy",
        "type": "Advanced Cost-Aware"
    },
    "mobilenetv2_continued_best": {
        "id": "mobilenetv2_continued_best",
        "name": "MobileNetV2 — Final Optimized Model",
        "description": "The ultimate best-performing model of the research pipeline. This is a fine-tuned iteration of the MobileNetV2+LDAM checkpoint, trained with a decaying cosine learning rate scheduler over extended epochs for maximum convergence and clinical reliability.",
        "weights_file": "mobilenetv2_continued_best.pth",
        "architecture": "MobileNetV2",
        "accuracy": "93.61% Test Accuracy / 93.89% Balanced Accuracy",
        "type": "Final Optimized Model"
    }
}

# Available Diffusion Models
DIFFUSION_MODELS = {
    "targeted_diffusion": {
        "id": "targeted_diffusion",
        "name": "Targeted Diffusion (SimpleUnet)",
        "description": "Generates synthetic lesion images for minority classes (MEL, BCC, AKIEC, BKL, VASC, DF)",
        "architecture": "SimpleUnet",
        "synthetic_images_url": "https://drive.google.com/drive/folders/1a2l9o7-YqyI9kDrkCNOxhYva695d6cRX?usp=drive_link"
    }
}

# Image preprocessing configuration
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# API Configuration
API_PREFIX = "/api"
DEBUG = True
PORT = 5000
