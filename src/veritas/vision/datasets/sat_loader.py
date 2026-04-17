"""General loader for SAT (Segment Anything for Medical Images) segmentation results.

This module provides utilities to load and access segmentation masks generated
by SAT models via the MCP integration.
"""

import json
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class SATSegmentationResult:
    """Container for a SAT segmentation result."""
    patient_id: str
    mask: np.ndarray  # Segmentation mask with integer labels
    spacing: tuple  # Voxel spacing in mm (e.g., (1.4, 1.4, 10.0))
    structures: List[str]  # Names of segmented structures
    metadata: Dict  # Full metadata from SAT
    mask_path: str  # Path to the NIfTI mask file
    model_name: str  # SAT model variant used
    timestamp: str  # When segmentation was created


def load_sat_manifest(results_db_path: Union[str, Path]) -> Dict:
    """Load the SAT segmentation manifest.

    The manifest.json file tracks all segmentation results in the database.

    Args:
        results_db_path: Path to the results_database directory

    Returns:
        Dictionary with manifest data including 'version', 'created', and 'results' list
    """
    results_db = Path(results_db_path)
    manifest_path = results_db / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    return manifest


def load_sat_result(
    results_db_path: Union[str, Path],
    patient_id: str,
    filter_fn: Optional[callable] = None,
    return_latest: bool = True
) -> SATSegmentationResult:
    """Load a SAT segmentation result for a patient.

    Args:
        results_db_path: Path to results_database directory
        patient_id: Patient ID (e.g., 'patient001')
        filter_fn: Optional function to filter results. Takes a result dict from
                  manifest and returns True if it should be included.
                  Example: lambda r: 'ED' in r['metadata'].get('source_image', '')
        return_latest: If multiple results match, return the most recent (default: True)

    Returns:
        SATSegmentationResult object

    Raises:
        ValueError: If no results found for patient_id
        FileNotFoundError: If mask file doesn't exist

    Example:
        >>> # Load any result for patient001
        >>> result = load_sat_result(results_db, "patient001")
        >>>
        >>> # Load result with custom filter (e.g., for specific cardiac phase)
        >>> result = load_sat_result(
        ...     results_db,
        ...     "patient001",
        ...     filter_fn=lambda r: 'frame01' in r.get('metadata', {}).get('source_image', '')
        ... )
        >>>
        >>> # Access the mask
        >>> print(f"Mask shape: {result.mask.shape}")
        >>> print(f"Structures: {result.structures}")
        >>> print(f"Spacing: {result.spacing}")
    """
    results_db = Path(results_db_path)
    manifest = load_sat_manifest(results_db)

    # Filter results for this patient
    patient_results = [
        r for r in manifest['results']
        if r['patient_id'] == patient_id
    ]

    if not patient_results:
        raise ValueError(f"No segmentation results found for patient '{patient_id}'")

    # Load metadata for each result to enable filtering
    results_with_metadata = []
    for result in patient_results:
        result_dir = Path(result['result_dir'])
        metadata_path = result_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            result_copy = result.copy()
            result_copy['metadata'] = metadata
            results_with_metadata.append(result_copy)
        else:
            result_copy = result.copy()
            result_copy['metadata'] = {}
            results_with_metadata.append(result_copy)

    # Apply custom filter if provided
    if filter_fn:
        filtered = [r for r in results_with_metadata if filter_fn(r)]
        if filtered:
            results_with_metadata = filtered
        else:
            print(f"Warning: No results matched filter for patient '{patient_id}'")

    # Select result (latest if multiple)
    if return_latest and len(results_with_metadata) > 1:
        # Sort by timestamp (descending)
        results_with_metadata.sort(key=lambda r: r['timestamp'], reverse=True)

    selected = results_with_metadata[0]

    # Load mask
    result_dir = Path(selected['result_dir'])
    mask_path = result_dir / "mask.nii.gz"

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    nii_img = nib.load(mask_path)
    mask = nii_img.get_fdata()
    spacing = nii_img.header.get_zooms()

    # Get metadata
    metadata = selected.get('metadata', {})

    return SATSegmentationResult(
        patient_id=patient_id,
        mask=mask,
        spacing=spacing,
        structures=metadata.get('structures', []),
        metadata=metadata,
        mask_path=str(mask_path),
        model_name=selected['model_name'],
        timestamp=selected['timestamp']
    )


def load_sat_results_batch(
    results_db_path: Union[str, Path],
    patient_ids: List[str],
    filter_fn: Optional[callable] = None,
    return_latest: bool = True
) -> List[SATSegmentationResult]:
    """Load SAT results for multiple patients.

    Args:
        results_db_path: Path to results_database directory
        patient_ids: List of patient IDs
        filter_fn: Optional filter function (see load_sat_result)
        return_latest: Return latest result if multiple match

    Returns:
        List of SATSegmentationResult objects

    Example:
        >>> patient_ids = ['patient001', 'patient002', 'patient003']
        >>> results = load_sat_results_batch(results_db, patient_ids)
        >>> for result in results:
        ...     print(f"{result.patient_id}: {result.mask.shape}")
    """
    results = []
    for patient_id in patient_ids:
        try:
            result = load_sat_result(results_db_path, patient_id, filter_fn, return_latest)
            results.append(result)
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Failed to load result for {patient_id}: {e}")
            continue

    return results


def list_patients_in_results(results_db_path: Union[str, Path]) -> List[str]:
    """List all unique patient IDs in the results database.

    Args:
        results_db_path: Path to results_database directory

    Returns:
        Sorted list of unique patient IDs
    """
    manifest = load_sat_manifest(results_db_path)
    patient_ids = sorted(set(r['patient_id'] for r in manifest['results']))
    return patient_ids


def calculate_volume(mask: np.ndarray, label_id: int, spacing: tuple) -> float:
    """Calculate volume of a specific label in the mask.

    Args:
        mask: Segmentation mask with integer labels
        label_id: Label ID to calculate volume for
        spacing: Voxel spacing in mm (e.g., (1.4, 1.4, 10.0))

    Returns:
        Volume in milliliters (mL)

    Example:
        >>> result = load_sat_result(results_db, "patient001")
        >>> # Assume label 3 is left ventricle
        >>> lv_volume_ml = calculate_volume(result.mask, label_id=3, spacing=result.spacing)
    """
    # Count voxels for this label
    label_voxels = np.sum(mask == label_id)

    # Calculate voxel volume in mm³
    voxel_volume_mm3 = np.prod(spacing)

    # Calculate total volume in mm³, then convert to mL
    volume_mm3 = label_voxels * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    return volume_ml


def get_unique_labels(mask: np.ndarray) -> np.ndarray:
    """Get unique label IDs present in the mask.

    Args:
        mask: Segmentation mask

    Returns:
        Array of unique label IDs

    Example:
        >>> result = load_sat_result(results_db, "patient001")
        >>> labels = get_unique_labels(result.mask)
        >>> print(f"Mask contains labels: {labels}")
    """
    return np.unique(mask)