#!/usr/bin/env python3
"""Generate dataset manifest for UCSF-PDGM (Preoperative Diffuse Glioma MRI) dataset.

Dataset structure:
    UCSF-PDGM/
    ├── UCSF-PDGM-metadata_v5.csv
    └── UCSF-PDGM-v5/
        └── UCSF-PDGM-XXXX_nifti/
            ├── UCSF-PDGM-XXXX_T1.nii.gz
            ├── UCSF-PDGM-XXXX_T1c.nii.gz
            ├── UCSF-PDGM-XXXX_T2.nii.gz
            ├── UCSF-PDGM-XXXX_FLAIR.nii.gz
            ├── UCSF-PDGM-XXXX_tumor_segmentation.nii.gz
            └── ... (additional modalities)

Metadata (16 columns):
    - ID, Sex, Age at MRI, WHO CNS Grade
    - Final pathologic diagnosis (WHO 2021)
    - MGMT status, MGMT index, 1p/19q, IDH
    - 1-dead 0-alive, OS (overall survival days)
    - EOR (extent of resection), Biopsy prior to imaging
    - BraTS21 ID, BraTS21 Segmentation Cohort, BraTS21 MGMT Cohort
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def simplify_idh_status(idh_value: str) -> str:
    """Convert detailed IDH status to simple wildtype/mutant."""
    if pd.isna(idh_value):
        return "unknown"
    idh_lower = str(idh_value).lower()
    if "wildtype" in idh_lower or "wild" in idh_lower:
        return "wildtype"
    elif "idh" in idh_lower or "mutated" in idh_lower:
        return "mutant"
    return "unknown"


def simplify_mgmt_status(mgmt_value: str) -> str:
    """Convert MGMT status to methylated/unmethylated/unknown."""
    if pd.isna(mgmt_value):
        return "unknown"
    mgmt_lower = str(mgmt_value).lower()
    if mgmt_lower == "positive":
        return "methylated"
    elif mgmt_lower == "negative":
        return "unmethylated"
    return "unknown"


def simplify_1p19q_status(value: str) -> str:
    """Convert 1p/19q status to codeleted/intact/unknown."""
    if pd.isna(value):
        return "unknown"
    val_lower = str(value).lower()
    if "co-deletion" in val_lower or "codeleted" in val_lower:
        return "codeleted"
    elif "intact" in val_lower:
        return "intact"
    return "unknown"


def generate_pdgm_manifest(
    data_root: Path,
    metadata_csv: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Generate dataset manifest for UCSF-PDGM.

    Args:
        data_root: Path to UCSF-PDGM folder
        metadata_csv: Path to metadata CSV (default: auto-detect)
        output_path: Where to save manifest JSON (default: data_root/dataset_manifest.json)

    Returns:
        Manifest dictionary
    """
    data_root = Path(data_root)
    output_path = output_path or data_root / "dataset_manifest.json"

    # Auto-detect metadata CSV
    if metadata_csv is None:
        csv_candidates = list(data_root.glob("*metadata*.csv"))
        if csv_candidates:
            metadata_csv = csv_candidates[0]
        else:
            raise FileNotFoundError("No metadata CSV found in data_root")

    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"Loaded metadata: {len(df)} patients, {len(df.columns)} columns")

    # Find image directory
    image_dir = data_root / "UCSF-PDGM-v5"
    if not image_dir.exists():
        # Try to find any folder with patient data
        candidates = [d for d in data_root.iterdir() if d.is_dir() and "PDGM" in d.name]
        if candidates:
            image_dir = candidates[0]
        else:
            print(f"Warning: Image directory not found at {image_dir}")

    samples = []
    missing_folders = 0

    for _, row in df.iterrows():
        patient_id = row["ID"]

        # Find patient folder (format: UCSF-PDGM-XXXX_nifti)
        # ID in CSV is like "UCSF-PDGM-004", folder is "UCSF-PDGM-0004_nifti"
        # Some IDs have suffixes like "UCSF-PDGM-0429_FU003d" - extract numeric part
        patient_num_str = patient_id.split("-")[-1]
        patient_num = "".join(c for c in patient_num_str if c.isdigit())
        if not patient_num:
            missing_folders += 1
            continue
        # Use 4-digit zero-padded ID matching actual NIfTI filenames
        canonical_id = f"UCSF-PDGM-{int(patient_num):04d}"
        folder_name = f"{canonical_id}_nifti"
        patient_dir = image_dir / folder_name

        if not patient_dir.exists():
            missing_folders += 1
            continue

        # Build observations dict for key modalities
        observations = {}
        modalities = {
            "T1": f"{folder_name.replace('_nifti', '')}_T1.nii.gz",
            "T1c": f"{folder_name.replace('_nifti', '')}_T1c.nii.gz",
            "T2": f"{folder_name.replace('_nifti', '')}_T2.nii.gz",
            "FLAIR": f"{folder_name.replace('_nifti', '')}_FLAIR.nii.gz",
        }

        for obs_name, filename in modalities.items():
            file_path = patient_dir / filename
            if file_path.exists():
                rel_path = file_path.relative_to(data_root)
                observations[obs_name] = {"path": str(rel_path)}

        # Check for tumor segmentation ground truth
        seg_filename = f"{folder_name.replace('_nifti', '')}_tumor_segmentation.nii.gz"
        seg_path = patient_dir / seg_filename
        has_gt = seg_path.exists()

        # Determine group based on grade
        grade = row["WHO CNS Grade"]
        if grade == 2:
            group = "GradeII"
        elif grade == 3:
            group = "GradeIII"
        else:
            group = "GradeIV"

        # Build metadata
        patient_metadata = {
            "age": int(row["Age at MRI"]) if pd.notna(row["Age at MRI"]) else None,
            "sex": row["Sex"] if pd.notna(row["Sex"]) else None,
            "grade": int(grade),
            "diagnosis": row["Final pathologic diagnosis (WHO 2021)"] if pd.notna(row["Final pathologic diagnosis (WHO 2021)"]) else None,
            "idh_status": simplify_idh_status(row["IDH"]),
            "idh_detail": row["IDH"] if pd.notna(row["IDH"]) else None,
            "mgmt_status": simplify_mgmt_status(row["MGMT status"]),
            "codeletion_1p19q": simplify_1p19q_status(row["1p/19q"]),
            "survival_status": "deceased" if row["1-dead 0-alive"] == 1 else "alive",
            "survival_days": float(row["OS"]) if pd.notna(row["OS"]) else None,
            "extent_of_resection": row["EOR"] if pd.notna(row["EOR"]) else None,
            "biopsy_prior": row["Biopsy prior to imaging"] == "Yes" if pd.notna(row["Biopsy prior to imaging"]) else None,
            "brats21_id": row["BraTS21 ID"] if pd.notna(row["BraTS21 ID"]) else None,
            "has_ground_truth": has_gt,
        }

        samples.append({
            "sample_id": canonical_id,
            "group": group,
            "observations": observations,
            "metadata": patient_metadata,
        })

    manifest = {
        "name": "ucsf_pdgm",
        "description": "UCSF Preoperative Diffuse Glioma MRI dataset",
        "version": "5.0",
        "source": "https://www.cancerimagingarchive.net/collection/ucsf-pdgm/",
        "groups": ["GradeII", "GradeIII", "GradeIV"],
        "observations": ["T1", "T1c", "T2", "FLAIR"],
        "samples": samples,
    }

    # Save manifest
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"\nManifest generated: {output_path}")
    print(f"Total samples: {len(samples)}")
    if missing_folders > 0:
        print(f"Missing folders (not yet transferred): {missing_folders}")

    # Group counts
    group_counts = {}
    idh_counts = {"wildtype": 0, "mutant": 0, "unknown": 0}
    for sample in samples:
        g = sample["group"]
        group_counts[g] = group_counts.get(g, 0) + 1
        idh = sample["metadata"]["idh_status"]
        idh_counts[idh] = idh_counts.get(idh, 0) + 1

    print("\nBy Grade:")
    for g in ["GradeII", "GradeIII", "GradeIV"]:
        print(f"  {g}: {group_counts.get(g, 0)}")

    print("\nBy IDH status:")
    for idh, count in idh_counts.items():
        print(f"  {idh}: {count}")

    return manifest


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from veritas.data_paths import PDGM_ROOT

    parser = argparse.ArgumentParser(description="Generate UCSF-PDGM manifest")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PDGM_ROOT,
        help="Path to UCSF-PDGM dataset",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for manifest JSON",
    )
    args = parser.parse_args()

    generate_pdgm_manifest(args.data_root, args.metadata, args.output)
