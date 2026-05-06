"""
ClassificationModel Domain Class (OOAD)
Represents a trained classification model for lesion inference
"""

class ClassificationModel:
    """Domain model representing a trained classification model"""
    
    def __init__(self, model_id: str, name: str, description: str,
                 architecture: str, weights_file: str, accuracy: float = None,
                 model_type: str = "baseline"):
        self._id = model_id
        self._name = name
        self._description = description
        self._architecture = architecture
        self._weights_file = weights_file
        self._accuracy = accuracy
        self._type = model_type
        self._loaded_model = None  # Lazy loading
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def architecture(self) -> str:
        return self._architecture
    
    @property
    def weights_file(self) -> str:
        return self._weights_file
    
    @property
    def accuracy(self) -> float:
        return self._accuracy
    
    @property
    def model_type(self) -> str:
        return self._type
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded_model is not None
    
    def set_loaded_model(self, model):
        """Set the loaded PyTorch model"""
        self._loaded_model = model
    
    def get_loaded_model(self):
        """Get the loaded PyTorch model"""
        return self._loaded_model
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON response"""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "architecture": self._architecture,
            "accuracy": self._accuracy,
            "type": self._type,
            "is_loaded": self.is_loaded
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ClassificationModel':
        """Create instance from dictionary"""
        return cls(
            model_id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            architecture=data.get("architecture"),
            weights_file=data.get("weights_file"),
            accuracy=data.get("accuracy"),
            model_type=data.get("type", "baseline")
        )
