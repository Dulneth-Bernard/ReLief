"""
PredictionResult Domain Class (OOAD)
Represents the result of a classification inference
"""
from typing import Dict, List
from datetime import datetime


class PredictionResult:
    """Domain model representing a prediction result from classification"""
    
    def __init__(self, image_id: str, model_id: str,
                 predicted_class: str, confidence: float,
                 class_probabilities: Dict[str, float],
                 prediction_time: datetime = None):
        self._image_id = image_id
        self._model_id = model_id
        self._predicted_class = predicted_class
        self._confidence = confidence
        self._class_probabilities = class_probabilities
        self._prediction_time = prediction_time or datetime.now()
    
    @property
    def image_id(self) -> str:
        return self._image_id
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def predicted_class(self) -> str:
        return self._predicted_class
    
    @property
    def confidence(self) -> float:
        return self._confidence
    
    @property
    def class_probabilities(self) -> Dict[str, float]:
        return self._class_probabilities
    
    @property
    def prediction_time(self) -> datetime:
        return self._prediction_time
    
    def get_top_k_predictions(self, k: int = 3) -> List[Dict]:
        """Get top k predictions sorted by probability"""
        sorted_probs = sorted(
            self._class_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"class": cls, "probability": prob}
            for cls, prob in sorted_probs[:k]
        ]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON response"""
        return {
            "image_id": self._image_id,
            "model_id": self._model_id,
            "predicted_class": self._predicted_class,
            "confidence": round(self._confidence, 4),
            "class_probabilities": {
                k: round(v, 4) for k, v in self._class_probabilities.items()
            },
            "top_predictions": self.get_top_k_predictions(3),
            "prediction_time": self._prediction_time.isoformat()
        }
