"""Ground truth segmentation populator for ablation studies.

Instead of running SAT inference, populate the results database with ground
truth masks from the dataset. This allows comparing pipeline performance with
perfect vs. predicted segmentations.

Supported datasets:
- ACDC: GT masks from *_gt.nii.gz files (labels: 0=bg, 1=RV, 2=Myo, 3=LV)
- UCSF-PDGM: GT masks from *_tumor_segmentation.nii.gz (BraTS: 1=NCR, 2=ED, 4=ET)

Usage: set use_ground_truth=True in experiment config. Phase 2A will call
populate_gt_results() instead of segment_identifiers(). Phase 2B is unchanged.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import nibabel as nib

from veritas.vision.results_db import ResultsDatabase
from veritas.vision.datasets.acdc_loader import ACDCDatasetLoader


# Maps SAT structure names (as written by agents) to ACDC GT label indices.
# ACDC GT: 0=background, 1=RV, 2=MYO, 3=LV
ACDC_STRUCTURE_LABEL_MAP: Dict[str, int] = {
    "left heart ventricle": 3,
    "left ventricle": 3,
    "lv": 3,
    "right heart ventricle": 1,
    "right ventricle": 1,
    "rv": 1,
    "myocardium": 2,
    "myo": 2,
}

# Maps SAT structure names to UCSF-PDGM BraTS GT label indices.
# BraTS GT: 0=background, 1=necrotic core (NCR), 2=peritumoral edema (ED), 4=enhancing tumor (ET)
# Multi-label structures (e.g., whole tumor) use lists of label indices.
PDGM_STRUCTURE_LABEL_MAP: Dict[str, Union[int, List[int]]] = {
    "whole tumor": [1, 2, 4],
    "brain tumor": [1, 2, 4],
    "enhancing tumor": 4,
    "enhancing brain tumor": 4,
    "contrast-enhancing tumor": 4,
    "tumor core": [1, 4],
    "necrotic core": 1,
    "necrotic brain tumor core": 1,
    "necrosis": 1,
    "edema": 2,
    "peritumoral edema": 2,
}


def _extract_binary_mask(
    gt_data: np.ndarray,
    label_spec: Union[int, List[int]],
) -> np.ndarray:
    """Extract a binary mask from a multi-label GT volume.

    Args:
        gt_data: Ground truth volume with integer labels.
        label_spec: Single label index or list of indices to include.

    Returns:
        Binary mask (uint8) where 1 = structure present.
    """
    if isinstance(label_spec, list):
        return np.isin(gt_data, label_spec).astype(np.uint8)
    return (gt_data == label_spec).astype(np.uint8)


def _populate_acdc(
    identifiers: List[str],
    structures: List[str],
    db: ResultsDatabase,
    dataset_path: str,
) -> tuple:
    """Populate results DB with ACDC ground truth masks."""
    loader = ACDCDatasetLoader(dataset_path)
    errors = []
    processed = 0

    for identifier in identifiers:
        parts = identifier.split(":")
        if len(parts) != 3:
            errors.append({"identifier": identifier, "error": f"Invalid format: {identifier!r}"})
            continue

        _, patient_id, observation = parts

        try:
            metadata = loader.get_patient_metadata(patient_id)
            obs_upper = observation.upper()
            if obs_upper == "ED":
                frame_num = metadata["ed_frame"]
            elif obs_upper == "ES":
                frame_num = metadata["es_frame"]
            else:
                raise ValueError(f"Unknown observation '{observation}' for ACDC (expected ED or ES)")

            split = loader.get_patient_split(patient_id)
            patient_dir = loader.data_root / split / patient_id
            sample_id = f"{patient_id}_frame{frame_num:02d}"
            image_path = patient_dir / f"{sample_id}.nii.gz"
            gt_path = patient_dir / f"{sample_id}_gt.nii.gz"

            if not gt_path.exists():
                raise FileNotFoundError(f"GT file not found: {gt_path}")

            gt_nii = nib.load(str(gt_path))
            gt_data = gt_nii.get_fdata().astype(np.uint8)

            structures_dir = Path(db.results_root) / f"seg_{sample_id}"
            structures_dir.mkdir(parents=True, exist_ok=True)

            for structure in structures:
                label_spec = ACDC_STRUCTURE_LABEL_MAP.get(structure.lower())
                if label_spec is None:
                    raise ValueError(
                        f"No GT label mapping for structure '{structure}'. "
                        f"Known: {list(ACDC_STRUCTURE_LABEL_MAP.keys())}"
                    )
                binary_mask = _extract_binary_mask(gt_data, label_spec)
                out_nii = nib.Nifti1Image(binary_mask, gt_nii.affine, gt_nii.header)
                out_nii.header.set_data_dtype(np.uint8)
                nib.save(out_nii, str(structures_dir / f"{structure}.nii.gz"))

            db.save_segmentation_result(
                patient_id=patient_id,
                model_name="ground_truth",
                mask_path=str(structures_dir),
                metadata={
                    "source_image": str(image_path),
                    "structures": structures,
                    "modality": "mri",
                    "model_variant": "ground_truth",
                    "cardiac_phase": obs_upper,
                },
            )
            processed += 1

        except Exception as e:
            errors.append({"identifier": identifier, "error": str(e)})

    return processed, errors


def _populate_pdgm(
    identifiers: List[str],
    structures: List[str],
    db: ResultsDatabase,
    dataset_path: str,
) -> tuple:
    """Populate results DB with UCSF-PDGM ground truth masks."""
    from veritas.vision.datasets.manifest_loader import ManifestDatasetLoader

    loader = ManifestDatasetLoader(data_root=dataset_path)
    errors = []
    processed = 0

    for identifier in identifiers:
        parts = identifier.split(":")
        if len(parts) != 3:
            errors.append({"identifier": identifier, "error": f"Invalid format: {identifier!r}"})
            continue

        _, patient_id, observation = parts

        try:
            # Resolve the source image path from the manifest
            source_image_path = loader.resolve_identifier(identifier)

            # Find the GT segmentation file: same directory, *_tumor_segmentation.nii.gz
            source_dir = Path(source_image_path).parent
            gt_candidates = list(source_dir.glob("*_tumor_segmentation.nii.gz"))
            if not gt_candidates:
                raise FileNotFoundError(
                    f"No GT segmentation found in {source_dir}. "
                    f"Expected *_tumor_segmentation.nii.gz"
                )
            gt_path = gt_candidates[0]

            gt_nii = nib.load(str(gt_path))
            gt_data = gt_nii.get_fdata().astype(np.uint8)

            # Use patient_id + observation for unique structures_dir
            seg_id = f"{patient_id}_{observation}"
            structures_dir = Path(db.results_root) / f"seg_{seg_id}"
            structures_dir.mkdir(parents=True, exist_ok=True)

            for structure in structures:
                label_spec = PDGM_STRUCTURE_LABEL_MAP.get(structure.lower())
                if label_spec is None:
                    raise ValueError(
                        f"No GT label mapping for structure '{structure}'. "
                        f"Known: {list(PDGM_STRUCTURE_LABEL_MAP.keys())}"
                    )
                binary_mask = _extract_binary_mask(gt_data, label_spec)
                out_nii = nib.Nifti1Image(binary_mask, gt_nii.affine, gt_nii.header)
                out_nii.header.set_data_dtype(np.uint8)
                nib.save(out_nii, str(structures_dir / f"{structure}.nii.gz"))

            db.save_segmentation_result(
                patient_id=patient_id,
                model_name="ground_truth",
                mask_path=str(structures_dir),
                metadata={
                    "source_image": str(source_image_path),
                    "structures": structures,
                    "modality": "mri",
                    "model_variant": "ground_truth",
                    "observation": observation,
                },
            )
            processed += 1

        except Exception as e:
            errors.append({"identifier": identifier, "error": str(e)})

    return processed, errors


def populate_gt_results(
    identifiers: List[str],
    structures: List[str],
    results_database: str,
    dataset_path: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """Populate results database with ground truth masks.

    Reads GT segmentation files from the dataset and registers them into
    the ResultsDatabase with model_name='ground_truth', using the same
    directory layout as SAT so that load_structure_mask() works unchanged.

    Args:
        identifiers: Dataset identifiers (e.g. ["acdc:patient001:ED", ...])
        structures: Structure names to extract (e.g. ["left heart ventricle"])
        results_database: Path to results database directory
        dataset_path: Path to dataset root directory
        dataset_name: Dataset name ("acdc" or "ucsf_pdgm")

    Returns:
        Summary dict matching the shape of segment_identifiers() output.
    """
    db = ResultsDatabase(results_database)
    name_lower = dataset_name.lower().replace("-", "_")

    if name_lower == "acdc":
        processed, errors = _populate_acdc(identifiers, structures, db, dataset_path)
    elif name_lower in ("ucsf_pdgm", "pdgm"):
        processed, errors = _populate_pdgm(identifiers, structures, db, dataset_path)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}' for GT population. "
            f"Supported: acdc, ucsf_pdgm"
        )

    success = len(errors) == 0
    if not success:
        print(f"  GT population: {len(errors)} error(s) out of {len(identifiers)} identifiers")
        for err in errors[:5]:
            print(f"    {err['identifier']}: {err['error']}")

    return {
        "success": success,
        "total_identifiers": len(identifiers),
        "structures": structures,
        "results_database": results_database,
        "processed_count": processed,
        "cached_count": 0,
        "errors": errors,
        "model": "ground_truth",
    }
