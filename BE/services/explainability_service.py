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

# Optional Cloudinary upload — uses same credentials as gallery route
try:
    import cloudinary
    import cloudinary.uploader
    _CLOUDINARY_AVAILABLE = True
except ImportError:
    _CLOUDINARY_AVAILABLE = False


class GradCAM:
    """XAI implementation for CNN visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        # Register PyTorch hooks to capture activations during forward pass,
        # and gradients during backward pass to mathematically trace heatmaps.
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None, method='gradcam'):
        """Generate Grad-CAM or Grad-CAM++ heatmap"""
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
        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]
        
        if method == 'gradcam++':
            gradients_sq = gradients.pow(2)
            gradients_cb = gradients.pow(3)
            
            # Sum activations over spatial dimensions
            activations_sum = activations.sum(dim=[1, 2], keepdim=True)
            
            alpha_num = gradients_sq
            alpha_denom = 2 * gradients_sq + activations_sum * gradients_cb
            
            # Avoid division by zero
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alphas = alpha_num / alpha_denom
            
            # Weights for each channel
            weights = (alphas * F.relu(gradients)).sum(dim=[1, 2])
        else:
            # Standard Grad-CAM
            weights = torch.mean(gradients, dim=[1, 2])
        
        # Apply weights to activations
        heatmap = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        heatmap = F.relu(heatmap).cpu().numpy()
        
        # Normalization
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
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
    
    def _upload_to_cloudinary(self, file_path: str, public_id: str) -> str:
        """Upload a local file to Cloudinary and return its secure URL.
        Returns None if credentials are missing or upload fails."""
        if not _CLOUDINARY_AVAILABLE:
            return None
        cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME')
        api_key = os.getenv('CLOUDINARY_API_KEY')
        api_secret = os.getenv('CLOUDINARY_API_SECRET')
        if not all([cloud_name, api_key, api_secret]):
            return None
        try:
            cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
            result = cloudinary.uploader.upload(
                file_path,
                public_id=f"relief_heatmaps/{public_id}",
                folder="relief_heatmaps",
                tags=["heatmap", "temp"],          # tag for easy bulk-delete later
                overwrite=True,
                resource_type="image"
            )
            return result.get('secure_url')
        except Exception as e:
            print(f"[ExplainabilityService] Cloudinary upload failed: {e}")
            return None

    def _get_target_layer(self, pytorch_model, model_id: str):
        """Return the correct last convolutional layer for Grad-CAM based on architecture."""
        from config import CLASSIFICATION_MODELS
        arch = CLASSIFICATION_MODELS.get(model_id, {}).get("architecture", "ResNet50").lower().replace("-", "").replace("_", "")
        if arch == "mobilenetv2":
            # Last conv block in MobileNetV2 features
            return pytorch_model.features[-1]
        elif arch == "efficientnetb0":
            # Last conv block in EfficientNet-B0 features
            return pytorch_model.features[-1]
        else:
            # ResNet50 - last residual block
            return pytorch_model.layer4[-1]

    def generate_gradcam(self, image_path: str, model_id: str,
                         target_class: str = None, method: str = 'gradcam') -> ExplainabilityResult:
        """Generate Grad-CAM explanation for an image"""
        # Load model
        classification_model = inference_service.load_model(model_id)
        pytorch_model = classification_model.get_loaded_model()
        
        # Get target layer based on architecture
        target_layer = self._get_target_layer(pytorch_model, model_id)
        
        # Create Grad-CAM
        grad_cam = GradCAM(pytorch_model, target_layer)
        
        # Preprocess image
        input_tensor = inference_service.preprocess_image(image_path)
        
        # Get target class index
        target_idx = None
        if target_class:
            target_idx = CLASS_NAMES.index(target_class)
        
        # Generate heatmap
        heatmap, predicted_idx = grad_cam.generate(input_tensor, target_idx, method)
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
        heatmap_filename = f"{image_id}_{model_id}_{method}_heatmap.png"
        overlay_filename = f"{image_id}_{model_id}_{method}_overlay.png"
        
        heatmap_path = os.path.join(self._heatmaps_dir, heatmap_filename)
        overlay_path = os.path.join(self._heatmaps_dir, overlay_filename)
        
        cv2.imwrite(heatmap_path, np.uint8(255 * heatmap_resized))
        cv2.imwrite(overlay_path, overlay)

        # Try uploading overlay to Cloudinary — store URL if successful
        overlay_public_id = f"{image_id}_{model_id}_{method}_overlay"
        cloudinary_url = self._upload_to_cloudinary(overlay_path, overlay_public_id)

        # Use Cloudinary URL when available, fall back to local static path
        final_overlay_url = cloudinary_url or f"/static/heatmaps/{overlay_filename}"
        final_heatmap_url = f"/static/heatmaps/{heatmap_filename}"

        if cloudinary_url:
            print(f"[ExplainabilityService] Overlay uploaded to Cloudinary: {cloudinary_url}")

        return ExplainabilityResult(
            image_id=image_id,
            model_id=model_id,
            target_class=actual_target_class,
            heatmap_path=final_heatmap_url,
            overlay_path=final_overlay_url,
            method=method
        )


# Singleton instance
explainability_service = ExplainabilityService()
