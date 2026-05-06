"""
InferenceService - Handles model loading and prediction
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODELS_DIR, CLASS_NAMES, CLASS_FULL_NAMES,
    CLASSIFICATION_MODELS, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
)
from models import ClassificationModel, PredictionResult


class InferenceService:
    """Service for loading models and running inference"""
    
    _instance = None
    _loaded_models: Dict[str, ClassificationModel] = {}
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    
    @property
    def device(self):
        return self._device
    
    def get_available_models(self) -> list:
        """Get list of available classification models"""
        return [
            ClassificationModel.from_dict(model_data).to_dict()
            for model_data in CLASSIFICATION_MODELS.values()
        ]
    
    def _build_architecture(self, architecture: str):
        """Build an untrained model with the correct output head for the given architecture."""
        num_classes = len(CLASS_NAMES)
        arch = architecture.lower().replace("-", "").replace("_", "")
        if arch == "mobilenetv2":
            model = models.mobilenet_v2(weights=None)
            # Custom head used during training:
            # classifier.0 = Dropout
            # classifier.1 = Linear(1280, 512)
            # classifier.2 = ReLU
            # classifier.3 = Dropout
            # classifier.4 = Linear(512, num_classes)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(model.last_channel, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(512, num_classes)
            )
        elif arch == "efficientnetb0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            # Default to ResNet50
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    def load_model(self, model_id: str) -> ClassificationModel:
        """Load a classification model by ID"""
        if model_id in self._loaded_models and self._loaded_models[model_id].is_loaded:
            return self._loaded_models[model_id]
        
        if model_id not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        
        model_config = CLASSIFICATION_MODELS[model_id]
        classification_model = ClassificationModel.from_dict(model_config)
        
        # Load PyTorch model
        weights_path = os.path.join(MODELS_DIR, model_config["weights_file"])
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        # Build the correct architecture based on config
        architecture = model_config.get("architecture", "ResNet50")
        pytorch_model = self._build_architecture(architecture)
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=self._device)
        pytorch_model.load_state_dict(state_dict)
        pytorch_model = pytorch_model.to(self._device)
        pytorch_model.eval()
        
        classification_model.set_loaded_model(pytorch_model)
        self._loaded_models[model_id] = classification_model
        
        return classification_model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for inference"""
        image = Image.open(image_path).convert('RGB')
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self._device)
    
    def predict(self, image_path: str, model_id: str) -> PredictionResult:
        """Run prediction on an image"""
        # Load model if not loaded
        classification_model = self.load_model(model_id)
        pytorch_model = classification_model.get_loaded_model()
        
        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            outputs = pytorch_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get prediction
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        # Build class probabilities dict
        class_probs = {
            CLASS_NAMES[i]: probabilities[i].item()
            for i in range(len(CLASS_NAMES))
        }
        
        # Extract image ID from path
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        
        return PredictionResult(
            image_id=image_id,
            model_id=model_id,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs
        )


# Singleton instance
inference_service = InferenceService()
