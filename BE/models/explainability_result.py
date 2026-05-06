"""
ExplainabilityResult Domain Class (OOAD)
Represents the result of explainability analysis (Grad-CAM heatmaps)
"""
from datetime import datetime
from typing import Optional


class ExplainabilityResult:
    """Domain model representing an explainability result (Grad-CAM heatmap)"""
    
    def __init__(self, image_id: str, model_id: str,
                 target_class: str, heatmap_path: str,
                 overlay_path: str = None,
                 method: str = "grad_cam",
                 created_at: datetime = None):
        self._image_id = image_id
        self._model_id = model_id
        self._target_class = target_class
        self._heatmap_path = heatmap_path
        self._overlay_path = overlay_path
        self._method = method
        self._created_at = created_at or datetime.now()
    
    @property
    def image_id(self) -> str:
        return self._image_id
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def target_class(self) -> str:
        return self._target_class
    
    @property
    def heatmap_path(self) -> str:
        return self._heatmap_path
    
    @property
    def overlay_path(self) -> Optional[str]:
        return self._overlay_path
    
    @property
    def method(self) -> str:
        return self._method
    
    @property
    def created_at(self) -> datetime:
        return self._created_at
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON response"""
        return {
            "image_id": self._image_id,
            "model_id": self._model_id,
            "target_class": self._target_class,
            "heatmap_path": self._heatmap_path,
            "overlay_path": self._overlay_path,
            "method": self._method,
            "created_at": self._created_at.isoformat()
        }
