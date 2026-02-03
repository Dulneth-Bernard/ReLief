"""
ExplainabilityService - Generates Grad-CAM heatmaps for model interpretability
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STATIC_DIR, CLASS_NAMES
from models import ExplainabilityResult
from services.inference_service import inference_service


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        
        return heatmap, target_class


class ExplainabilityService:
    """Service for generating model explanations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._heatmaps_dir = os.path.join(STATIC_DIR, "heatmaps")
        os.makedirs(self._heatmaps_dir, exist_ok=True)
    
    def generate_gradcam(self, image_path: str, model_id: str,
                         target_class: str = None) -> ExplainabilityResult:
        """Generate Grad-CAM explanation for an image"""
        # Load model
        classification_model = inference_service.load_model(model_id)
        pytorch_model = classification_model.get_loaded_model()
        
        # Get target layer (last conv layer for ResNet50)
        target_layer = pytorch_model.layer4[-1]
        
        # Create Grad-CAM
        grad_cam = GradCAM(pytorch_model, target_layer)
        
        # Preprocess image
        input_tensor = inference_service.preprocess_image(image_path)
        
        # Get target class index
        target_idx = None
        if target_class:
            target_idx = CLASS_NAMES.index(target_class)
        
        # Generate heatmap
        heatmap, predicted_idx = grad_cam.generate(input_tensor, target_idx)
        actual_target_class = CLASS_NAMES[predicted_idx]
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        
        # Create overlay
        overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)
        
        # Save files
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        heatmap_filename = f"{image_id}_{model_id}_heatmap.png"
        overlay_filename = f"{image_id}_{model_id}_overlay.png"
        
        heatmap_path = os.path.join(self._heatmaps_dir, heatmap_filename)
        overlay_path = os.path.join(self._heatmaps_dir, overlay_filename)
        
        cv2.imwrite(heatmap_path, np.uint8(255 * heatmap_resized))
        cv2.imwrite(overlay_path, overlay)
        
        return ExplainabilityResult(
            image_id=image_id,
            model_id=model_id,
            target_class=actual_target_class,
            heatmap_path=f"/static/heatmaps/{heatmap_filename}",
            overlay_path=f"/static/heatmaps/{overlay_filename}",
            method="grad_cam"
        )


# Singleton instance
explainability_service = ExplainabilityService()
