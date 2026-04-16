"""
ImageService - Handles image browsing and uploads
"""
import os
import sys
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UPLOADS_DIR, DIFFUSION_MODELS
from models import LesionImage, DiffusionModel


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class ImageService:
    """Service for handling image operations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    def get_diffusion_models(self) -> List[dict]:
        """Get list of available diffusion models"""
        return [
            DiffusionModel.from_dict(model_data).to_dict()
            for model_data in DIFFUSION_MODELS.values()
        ]
    
    def get_diffusion_model(self, model_id: str) -> Optional[DiffusionModel]:
        """Get a specific diffusion model by ID"""
        if model_id in DIFFUSION_MODELS:
            return DiffusionModel.from_dict(DIFFUSION_MODELS[model_id])
        return None
    
    def upload_image(self, file) -> LesionImage:
        """Upload an image file"""
        if not file or not file.filename:
            raise ValueError("No file provided")
        
        if not allowed_file(file.filename):
            raise ValueError(f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")
        
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        extension = original_filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        new_filename = f"{unique_id}_{original_filename}"
        
        # Save file
        filepath = os.path.join(UPLOADS_DIR, new_filename)
        file.save(filepath)
        
        return LesionImage(
            image_id=unique_id,
            filename=new_filename,
            path=filepath,
            source="uploaded",
            uploaded_at=datetime.now()
        )
    
    def get_uploaded_images(self) -> List[LesionImage]:
        """Get list of uploaded images"""
        images = []
        if os.path.exists(UPLOADS_DIR):
            for filename in os.listdir(UPLOADS_DIR):
                if allowed_file(filename):
                    filepath = os.path.join(UPLOADS_DIR, filename)
                    images.append(LesionImage.from_file(filepath, source="uploaded"))
        return images
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """Get the full path for an image by ID"""
        # 1. Check uploads directory
        if os.path.exists(UPLOADS_DIR):
            for filename in os.listdir(UPLOADS_DIR):
                if image_id in filename:
                    return os.path.join(UPLOADS_DIR, filename)
                    
        # 2. Check sample-images directory
        from config import STATIC_DIR
        samples_dir = os.path.join(STATIC_DIR, "sample-images")
        if os.path.exists(samples_dir):
            for filename in os.listdir(samples_dir):
                if image_id in filename:
                    return os.path.join(samples_dir, filename)
                    
        return None


# Singleton instance
image_service = ImageService()
