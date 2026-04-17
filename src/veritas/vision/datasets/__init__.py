"""Medical dataset loaders."""

from ..base import MedicalDatasetLoader, MedicalImage, DatasetRegistry
from .acdc_loader import ACDCDatasetLoader
from .manifest_loader import ManifestDatasetLoader
from .sat_loader import (
    load_sat_manifest,
    load_sat_result,
    load_sat_results_batch,
    list_patients_in_results,
    calculate_volume,
    get_unique_labels,
    SATSegmentationResult,
)

# Auto-register ACDC dataset
DatasetRegistry.register("ACDC", ACDCDatasetLoader)

__all__ = [
    "MedicalDatasetLoader",
    "MedicalImage",
    "DatasetRegistry",
    "ACDCDatasetLoader",
    "ManifestDatasetLoader",
    "load_sat_manifest",
    "load_sat_result",
    "load_sat_results_batch",
    "list_patients_in_results",
    "calculate_volume",
    "get_unique_labels",
    "SATSegmentationResult",
]
