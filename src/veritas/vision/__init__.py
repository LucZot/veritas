"""
Vision utilities for VERITAS.

This module provides:
- Medical dataset loaders (ACDC cardiac MRI, UCSF-PDGM glioma)
- Metrics computation (Dice, IoU, volumes, ejection fraction)
- Visualization tools (overlays, comparisons, animations)

Vision models (e.g., SAT) are accessed via MCP servers for better isolation.
"""

from .base import (
    # Data structures
    MedicalImage,

    # Abstract interfaces
    MedicalDatasetLoader,

    # Registries
    DatasetRegistry,
)

from .results_db import (
    ResultsDatabase,
)

__all__ = [
    "MedicalImage",
    "MedicalDatasetLoader",
    "DatasetRegistry",
    "ResultsDatabase",
]

__version__ = "0.1.0"  # Vision module version
