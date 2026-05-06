"""
LesionImage Domain Class (OOAD)
Represents a dermoscopic lesion image
"""
from datetime import datetime
from typing import Optional


class LesionImage:
    """Domain model representing a dermoscopic lesion image"""
    
    def __init__(self, image_id: str, filename: str, path: str,
                 source: str = "isic", label: str = None,
                 uploaded_at: datetime = None):
        self._id = image_id
        self._filename = filename
        self._path = path
        self._source = source  # 'isic', 'synthetic', 'uploaded'
        self._label = label
        self._uploaded_at = uploaded_at or datetime.now()
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def filename(self) -> str:
        return self._filename
    
    @property
    def path(self) -> str:
        return self._path
    
    @property
    def source(self) -> str:
        return self._source
    
    @property
    def label(self) -> Optional[str]:
        return self._label
    
    @property
    def uploaded_at(self) -> datetime:
        return self._uploaded_at
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON response"""
        return {
            "id": self._id,
            "filename": self._filename,
            "path": self._path,
            "source": self._source,
            "label": self._label,
            "uploaded_at": self._uploaded_at.isoformat() if self._uploaded_at else None
        }
    
    @classmethod
    def from_file(cls, filepath: str, source: str = "isic", 
                  label: str = None) -> 'LesionImage':
        """Create instance from file path"""
        import os
        filename = os.path.basename(filepath)
        image_id = os.path.splitext(filename)[0]
        return cls(
            image_id=image_id,
            filename=filename,
            path=filepath,
            source=source,
            label=label
        )
