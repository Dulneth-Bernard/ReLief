"""
DiffusionModel Domain Class (OOAD)
Represents a diffusion model architecture for synthetic image generation
"""

class DiffusionModel:
    """Domain model representing a diffusion architecture for synthetic data generation"""
    
    def __init__(self, model_id: str, name: str, description: str, 
                 architecture: str, synthetic_images_url: str = None):
        self._id = model_id
        self._name = name
        self._description = description
        self._architecture = architecture
        self._synthetic_images_url = synthetic_images_url
    
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
    def synthetic_images_url(self) -> str:
        return self._synthetic_images_url
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON response"""
        return {
            "id": self._id,
            "name": self._name,
            "description": self._description,
            "architecture": self._architecture,
            "synthetic_images_url": self._synthetic_images_url
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DiffusionModel':
        """Create instance from dictionary"""
        return cls(
            model_id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            architecture=data.get("architecture"),
            synthetic_images_url=data.get("synthetic_images_url")
        )
