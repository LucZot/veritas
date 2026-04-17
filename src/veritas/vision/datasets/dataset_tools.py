"""LangChain tools for querying medical imaging dataset metadata.

These tools enable agents to discover patient information without
hardcoding frame numbers or paths. Agents can query:
- Individual patient metadata (observations, identifiers, clinical data)
- Lists of patients filtered by group
- Available groups/cohorts in a dataset
- Available observations (imaging modalities/timepoints)
- Resolve identifiers to file paths

Identifier format (used across the workflow):
- Dataset identifiers are strings like "acdc:patient001:ED"
- In segmentation_request.json, `identifiers` is a list of these strings
- To recover patient_id: `identifier.split(":")[1]`
- Use `get_dataset_patient_info(...).get("group")` or
  `sat.get_patient_metadata(patient_id)["group"]` for cohort labels

Currently supports:
- ACDC: Automated Cardiac Diagnosis Challenge dataset (cardiac MRI)
- UCSF-PDGM: Diffuse Glioma MRI dataset (brain tumor)
- Any dataset with a dataset manifest file

Environment variables for dataset roots:
- DATASET_PATH: Default path for all datasets
- ACDC_DATA_ROOT or ACDC_ROOT: ACDC-specific path
- PDGM_DATA_ROOT or UCSF_PDGM_ROOT: UCSF-PDGM specific path

This supports autonomous agent workflows where agents construct
segmentation requests based on discovered metadata.

Example workflow:
    1. list_dataset_patients("acdc", group="DCM") → get patient IDs
    2. get_dataset_patient_info("acdc", patient_id) → get ED/ES paths
    3. segment_structures_batch(image_paths=[...]) → segment all frames
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from veritas.vision.datasets.acdc_loader import ACDCDatasetLoader
from veritas.vision.datasets.manifest_loader import ManifestDatasetLoader

# Dataset loader registry (singleton pattern)
_dataset_loaders: Dict[str, Any] = {}

# Dataset root registry - maps dataset names to their root paths
_dataset_roots: Dict[str, str] = {}


def register_dataset_root(dataset: str, root_path: str) -> None:
    """Register a dataset root path for a specific dataset.

    Args:
        dataset: Dataset name (e.g., "acdc", "ucsf_pdgm")
        root_path: Path to dataset root directory
    """
    _dataset_roots[dataset.lower()] = root_path


def _get_dataset_root(dataset: str = "acdc") -> str:
    """Resolve dataset root from registry or environment.

    Priority order:
    1. Registered root via register_dataset_root()
    2. Dataset-specific environment variable
    3. DATASET_PATH environment variable

    Args:
        dataset: Dataset name to resolve root for

    Returns:
        Path to dataset root directory

    Raises:
        ValueError: If no root can be resolved
    """
    dataset_lower = dataset.lower()

    # Check registered roots first
    if dataset_lower in _dataset_roots:
        return _dataset_roots[dataset_lower]

    # Dataset-specific environment variables
    env_mappings = {
        "acdc": ["ACDC_DATA_ROOT", "ACDC_ROOT"],
        "ucsf_pdgm": ["PDGM_DATA_ROOT", "UCSF_PDGM_ROOT"],
        "ucsf-pdgm": ["PDGM_DATA_ROOT", "UCSF_PDGM_ROOT"],
    }

    # Check dataset-specific env vars
    for env_var in env_mappings.get(dataset_lower, []):
        if os.environ.get(env_var):
            return os.environ[env_var]

    # Fall back to generic DATASET_PATH
    if os.environ.get("DATASET_PATH"):
        return os.environ["DATASET_PATH"]

    raise ValueError(
        f"Dataset root not configured for '{dataset}'. "
        f"Set DATASET_PATH or use register_dataset_root()."
    )


def _get_acdc_loader() -> ACDCDatasetLoader:
    """Get or create ACDC loader instance (singleton pattern)."""
    if "acdc" not in _dataset_loaders:
        _dataset_loaders["acdc"] = ACDCDatasetLoader(
            data_root=_get_dataset_root("acdc"),
            split="all"
        )
    return _dataset_loaders["acdc"]


def _get_manifest_loader(dataset: str) -> ManifestDatasetLoader:
    """Get or create a manifest-based loader for non-ACDC datasets."""
    key = f"manifest:{dataset.lower()}"
    if key not in _dataset_loaders:
        _dataset_loaders[key] = ManifestDatasetLoader(
            data_root=_get_dataset_root(dataset),
            dataset_name=dataset
        )
    return _dataset_loaders[key]


def _get_loader(dataset: str):
    """Get appropriate dataset loader."""
    dataset_lower = dataset.lower()
    if dataset_lower == "acdc":
        return _get_acdc_loader()
    return _get_manifest_loader(dataset)


@tool
def get_dataset_patient_info(dataset: str, patient_id: str) -> dict:
    """Get metadata for a patient in a medical imaging dataset.

    This tool reads dataset-specific metadata files to retrieve:
    - Temporal frame information (e.g., ED/ES for cardiac imaging)
    - Patient group label (e.g., disease group, grade)
    - Demographics (height, weight, etc.)
    - Dataset-relative identifiers (NOT full file paths)

    Use this tool when you need to segment a specific patient but don't
    know which frame numbers to use. The identifiers can be used directly
    with segmentation tools.

    Args:
        dataset: Dataset name (e.g., "acdc" or a manifest-defined dataset name)
        patient_id: Patient identifier (e.g., "patient001", "patient015")

    Returns:
        Dictionary with dataset-specific keys. For ACDC:
        - patient_id: Patient ID (str)
        - group: Group code (str) - e.g. NOR, DCM, HCM, MINF, RV for ACDC
        - ed_frame: ED frame number (int)
        - es_frame: ES frame number (int)
        - num_frames: Total frames in cardiac cycle (int)
        - height: Patient height in cm (float)
        - weight: Patient weight in kg (float)
        - ed_identifier: Dataset-relative identifier for ED frame (str)
        - es_identifier: Dataset-relative identifier for ES frame (str)

        For manifest datasets, returns:
        - patient_id: Patient ID (str)
        - group: Cohort/group label (str)
        - num_frames: Number of observations (int)
        - observations: Observation identifiers (list[str])
        - identifiers: Observation identifiers mapped to dataset identifiers (dict)
        - metadata: Additional metadata (dict)

    Example:
        >>> # Query ACDC patient
        >>> info = get_dataset_patient_info("acdc", "patient015")
        >>> print(f"Patient {info['patient_id']} is in group {info['group']}")
        Patient patient015 is in group DCM
        >>> print(f"ES identifier: {info['es_identifier']}")
        ES identifier: acdc:patient015:ES

        # Use in segmentation workflow:
        >>> info = get_dataset_patient_info("acdc", "patient015")
        >>> segment_medical_structure(
        ...     image_path=info['es_identifier'],
        ...     structures=['left heart ventricle', 'myocardium'],
        ...     modality='mri'
        ... )

    Raises:
        ValueError: If dataset manifest is invalid
        FileNotFoundError: If patient doesn't exist in dataset
    """
    loader = _get_loader(dataset)
    return loader.get_patient_metadata(patient_id)


@tool
def list_dataset_patients(
    dataset: str,
    group: Optional[str] = None,
    metadata_filters: Optional[Dict[str, str]] = None,
    include_patient_ids: bool = False,
) -> dict:
    """List all patients in a medical imaging dataset, optionally filtered by group and/or metadata.

    Use this tool to discover how many patients are available for analysis.
    Returns total_count by default. Set include_patient_ids=True to get the full list.

    Args:
        dataset: Dataset name (e.g., "acdc" or "ucsf_pdgm")
        group: Optional group filter (e.g., "DCM", "GradeIV").
               For ACDC: "NOR", "DCM", "HCM", "MINF", "RV"
               For UCSF-PDGM: "GradeII", "GradeIII", "GradeIV"
        metadata_filters: Optional dict to filter by patient metadata fields.
               For example: {"idh_status": "wildtype"} or {"extent_of_resection": "GTR"}.
               Can be combined with group, e.g., group="GradeIV" + metadata_filters={"idh_status": "mutant"}
               to get Grade IV IDH-mutant patients.
        include_patient_ids: If True, include full patient list. Default False (count only).

    Returns:
        Dictionary with:
        - dataset: Dataset name (str)
        - total_count: Number of patients returned (int)
        - filter: Applied group filter (str or None)
        - metadata_filters: Applied metadata filters (dict or None)
        - patients: (only if include_patient_ids=True) List of patient dicts

    Example:
        >>> # Get count of DCM patients from ACDC
        >>> result = list_dataset_patients("acdc", group="DCM")

        >>> # Get IDH-wildtype patients across all grades
        >>> result = list_dataset_patients("ucsf_pdgm", metadata_filters={"idh_status": "wildtype"})

        >>> # Get full patient list for code execution
        >>> result = list_dataset_patients("acdc", group="DCM", include_patient_ids=True)

    Raises:
        ValueError: If dataset is not supported
    """
    loader = _get_loader(dataset)
    patients = loader.list_patients_by_group(group=group, metadata_filters=metadata_filters)

    result = {
        "dataset": dataset,
        "total_count": len(patients),
        "filter": group,
        "metadata_filters": metadata_filters,
    }
    if include_patient_ids:
        result["patients"] = patients
    return result


@tool
def resolve_dataset_identifier(identifier: str) -> dict:
    """Resolve a dataset identifier to its full file path.

    Dataset identifiers follow the format: "dataset:patient_id:observation"
    (e.g., "acdc:patient001:ED" or "ucsf_pdgm:UCSF-PDGM-004:T1c")

    Args:
        identifier: Dataset identifier string in format "dataset:patient:observation"

    Returns:
        Dictionary with identifier, file_path, dataset, patient_id, observation

    Raises:
        ValueError: If identifier format is invalid
        FileNotFoundError: If patient or observation not found
    """
    parts = identifier.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid identifier format: '{identifier}'. "
            "Expected 'dataset:patient_id:observation'"
        )

    dataset, patient_id, observation = parts
    loader = _get_loader(dataset)

    # ManifestDatasetLoader has resolve_identifier method
    if hasattr(loader, "resolve_identifier"):
        file_path = loader.resolve_identifier(identifier)
    else:
        # ACDC loader - construct path manually
        metadata = loader.get_patient_metadata(patient_id)
        if observation.upper() == "ED":
            frame_num = metadata["ed_frame"]
        elif observation.upper() == "ES":
            frame_num = metadata["es_frame"]
        else:
            raise ValueError(f"Unknown observation '{observation}' for ACDC")

        split = loader.get_patient_split(patient_id)
        file_path = str(
            loader.data_root / split / patient_id /
            f"{patient_id}_frame{frame_num:02d}.nii.gz"
        )

    return {
        "identifier": identifier,
        "file_path": file_path,
        "dataset": dataset,
        "patient_id": patient_id,
        "observation": observation,
    }
