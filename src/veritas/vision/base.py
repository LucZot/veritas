"""Base interfaces for vision foundation models and medical datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MedicalImage:
    """Standardized medical image container."""

    image_id: str                # Unique identifier
    data: np.ndarray            # Image data (H, W) or (H, W, D) for 3D
    pixel_spacing: Tuple[float, ...]  # Physical spacing (mm/pixel)
    modality: str               # "MRI", "CT", "X-ray", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # Dataset-specific metadata
    ground_truth: Optional[np.ndarray] = None  # GT masks if available

    def to_pil(self, slice_idx: Optional[int] = None):
        """Convert to PIL Image (RGB) for foundation model input."""
        from PIL import Image

        if self.data.ndim == 3 and slice_idx is not None:
            # 3D volume → extract slice
            slice_data = self.data[:, :, slice_idx]
        elif self.data.ndim == 2:
            slice_data = self.data
        else:
            raise ValueError(f"Cannot convert {self.data.ndim}D data without slice_idx")

        # Normalize to 0-255
        normalized = ((slice_data - slice_data.min()) /
                     (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)

        return Image.fromarray(normalized).convert("RGB")


class MedicalDatasetLoader(ABC):
    """Abstract base class for medical dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name (e.g., 'ACDC', 'AMOS', 'Custom')."""
        pass

    @property
    @abstractmethod
    def supported_structures(self) -> List[str]:
        """List of anatomical structures this dataset contains."""
        pass

    @abstractmethod
    def load_image(self, image_id: str, **kwargs) -> MedicalImage:
        """
        Load a medical image by ID.

        Args:
            image_id: Unique image identifier (dataset-specific format)
            **kwargs: Loader-specific parameters

        Returns:
            MedicalImage object with standardized format
        """
        pass

    @abstractmethod
    def list_available_images(self) -> List[str]:
        """Return list of all available image IDs in dataset."""
        pass

    @abstractmethod
    def get_image_metadata(self, image_id: str) -> Dict[str, Any]:
        """Get metadata for an image (patient info, acquisition params, etc.)."""
        pass


class DatasetRegistry:
    """Registry for available medical datasets."""

    _datasets: Dict[str, type[MedicalDatasetLoader]] = {}

    @classmethod
    def register(cls, name: str, dataset_class: type[MedicalDatasetLoader]):
        """Register a dataset implementation."""
        cls._datasets[name] = dataset_class

    @classmethod
    def get_dataset(cls, name: str, **init_kwargs) -> MedicalDatasetLoader:
        """Instantiate a registered dataset by name."""
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not registered. Available: {available}")
        return cls._datasets[name](**init_kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls._datasets.keys())
