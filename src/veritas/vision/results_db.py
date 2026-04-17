"""Simple file-based results management for segmentation outputs.

This module provides functions to organize, save, and retrieve segmentation
results, enabling agents to:
- Cache expensive segmentation computations
- Track experimental results across parameter sweeps
- Query previous results for analysis
- Avoid re-running segmentations

Architecture: File-based with manifest JSON for indexing.
Future: Can migrate to SQLite if needed for complex queries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import shutil


class ResultsDatabase:
    """
    File-based results storage with manifest indexing.

    Directory structure:
        results_root/
            patient_001/
                sat_nano_2024-10-30_14-23/
                    mask.nii.gz
                    metrics.json
                    overlay.png
                    report.md
                sat_pro_2024-10-30_15-45/
                    ...
            patient_002/
                ...
            manifest.json  # Index of all results
    """

    def __init__(self, results_root: str):
        """
        Initialize results database.

        Args:
            results_root: Root directory for storing results
        """
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.results_root / "manifest.json"

        # Load or create manifest
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "results": []
            }
            self._save_manifest()

    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def save_segmentation_result(
        self,
        patient_id: str,
        model_name: str,
        mask_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        metrics_path: Optional[str] = None,
        visualizations: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Save segmentation result with organized file structure.

        Args:
            patient_id: Patient identifier
            model_name: Model/method name (e.g., "sat_nano", "sat_pro")
            mask_path: Path to segmentation mask NIfTI file
            metadata: Additional metadata (parameters, timestamp, etc.)
            metrics_path: Optional path to metrics JSON file
            visualizations: Optional dict of {name: path} for PNG files

        Returns:
            Result ID (directory name) for later retrieval

        Example:
            >>> db = ResultsDatabase("/results")
            >>> result_id = db.save_segmentation_result(
            ...     "patient001",
            ...     "sat_nano",
            ...     "/tmp/seg.nii.gz",
            ...     metadata={"variant": "nano", "threshold": 0.5}
            ... )
        """
        # Create lightweight result directory (no copying, just references to SAT outputs)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_id = f"{model_name}_{timestamp}"

        # Determine structures directory from mask_path
        # SAT outputs individual structure masks in a directory (e.g., seg_patient001_frame01/)
        mask_path_obj = Path(mask_path).absolute()

        if mask_path_obj.is_dir():
            # mask_path is already the structures directory (new behavior)
            structures_dir = str(mask_path_obj)
        else:
            # Legacy case: mask_path points to a file
            # Try to derive structures directory by removing .nii.gz extension
            mask_path_str = str(mask_path_obj)
            if mask_path_str.endswith('.nii.gz'):
                potential_dir = mask_path_str[:-7]  # Remove '.nii.gz'
                if Path(potential_dir).exists() and Path(potential_dir).is_dir():
                    structures_dir = potential_dir
                else:
                    structures_dir = None
            else:
                structures_dir = None

        # Build minimal manifest entry with reference to SAT output
        manifest_entry = {
            "patient_id": patient_id,
            "model_name": model_name,
            "result_id": result_id,
            "timestamp": timestamp,
            "structures_dir": structures_dir  # Reference to individual structure masks directory
        }

        # Add source_image and other metadata if available
        if metadata:
            if "source_image" in metadata:
                manifest_entry["source_image"] = metadata["source_image"]
            if "structures" in metadata:
                manifest_entry["structures"] = metadata["structures"]
            # Add any other relevant metadata fields
            for key in ["modality", "model_variant", "cardiac_phase", "processing_time"]:
                if key in metadata:
                    manifest_entry[key] = metadata[key]

        self.manifest["results"].append(manifest_entry)
        self._save_manifest()

        return result_id

    def get_latest_result(
        self,
        patient_id: str,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve most recent segmentation result for a patient.

        Args:
            patient_id: Patient identifier
            model_name: Optional filter by model name

        Returns:
            Metadata dict or None if not found

        Example:
            >>> db = ResultsDatabase("/results")
            >>> result = db.get_latest_result("patient001", "sat_nano")
            >>> if result:
            ...     print(f"Mask: {result['mask_path']}")
        """
        # Filter manifest
        matching = [
            r for r in self.manifest["results"]
            if r["patient_id"] == patient_id and
            (model_name is None or r["model_name"] == model_name)
        ]

        if not matching:
            return None

        # Get most recent
        latest = max(matching, key=lambda x: x["timestamp"])

        # Load full metadata
        metadata_path = Path(latest["result_dir"]) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return latest

    def get_result_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific result by ID.

        Args:
            result_id: Result identifier (e.g., "sat_nano_2024-10-30_14-23")

        Returns:
            Metadata dict or None if not found
        """
        # Find in manifest
        matching = [
            r for r in self.manifest["results"]
            if r["result_id"] == result_id
        ]

        if not matching:
            return None

        # Load full metadata
        metadata_path = Path(matching[0]["result_dir"]) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return matching[0]

    def list_patient_results(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        List all results for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of result metadata dicts sorted by timestamp (newest first)

        Example:
            >>> db = ResultsDatabase("/results")
            >>> results = db.list_patient_results("patient001")
            >>> for r in results:
            ...     print(f"{r['model_name']}: {r['timestamp']}")
        """
        matching = [
            r for r in self.manifest["results"]
            if r["patient_id"] == patient_id
        ]

        # Sort by timestamp (newest first)
        matching.sort(key=lambda x: x["timestamp"], reverse=True)

        # Load full metadata for each
        full_results = []
        for entry in matching:
            metadata_path = Path(entry["result_dir"]) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    full_results.append(json.load(f))
            else:
                full_results.append(entry)

        return full_results

    def delete_result(self, result_id: str) -> bool:
        """
        Delete a result from database.

        Args:
            result_id: Result identifier to delete

        Returns:
            True if deleted, False if not found
        """
        # Find in manifest
        matching_idx = None
        result_dir = None
        for idx, r in enumerate(self.manifest["results"]):
            if r["result_id"] == result_id:
                matching_idx = idx
                result_dir = Path(r["result_dir"])
                break

        if matching_idx is None:
            return False

        # Delete directory
        if result_dir and result_dir.exists():
            shutil.rmtree(result_dir)

        # Remove from manifest
        del self.manifest["results"][matching_idx]
        self._save_manifest()

        return True

    def result_exists(
        self, 
        patient_id: str, 
        model_name: str, 
        structures: Optional[List[str]] = None
    ) -> bool:
        """
        Check if a segmentation result already exists.

        Args:
            patient_id: Patient identifier
            model_name: Model/method name
            structures: Optional list of structures to check for

        Returns:
            True if result exists, False otherwise

        Example:
            >>> db = ResultsDatabase("/results")
            >>> if db.result_exists("patient001", "sat_nano", ["left heart ventricle"]):
            ...     print("Result already exists, skipping inference")
        """
        matching = [
            r for r in self.manifest["results"]
            if r["patient_id"] == patient_id and r["model_name"] == model_name
        ]

        if not matching:
            return False

        # If structures are specified, check if they match
        if structures:
            for result in matching:
                # Load metadata to check structures
                metadata_path = Path(result["result_dir"]) / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        result_structures = metadata.get("structures", [])
                        # Check if all requested structures are present
                        if all(struct in result_structures for struct in structures):
                            return True
                return False

        return True

    def find_existing_result(
        self,
        patient_id: str,
        model_name: str,
        structures: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find an existing result that matches the criteria.

        Args:
            patient_id: Patient identifier  
            model_name: Model/method name
            structures: Optional list of structures to match

        Returns:
            Matching result metadata or None

        Example:
            >>> db = ResultsDatabase("/results")
            >>> existing = db.find_existing_result("patient001", "sat_nano", ["myocardium"])
            >>> if existing:
            ...     print(f"Found existing result: {existing['result_id']}")
        """
        matching = [
            r for r in self.manifest["results"]
            if r["patient_id"] == patient_id and r["model_name"] == model_name
        ]

        if not matching:
            return None

        # If structures specified, find best match
        if structures:
            for result in matching:
                metadata_path = Path(result["result_dir"]) / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        result_structures = metadata.get("structures", [])
                        # Check if all requested structures are present
                        if all(struct in result_structures for struct in structures):
                            return metadata
            return None

        # Return most recent if no structure constraint
        latest = max(matching, key=lambda x: x["timestamp"])
        metadata_path = Path(latest["result_dir"]) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return latest

    def get_missing_patients(
        self,
        patient_list: List[str],
        model_name: str,
        structures: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of patients that don't have results for specified model/structures.

        Args:
            patient_list: List of patient IDs to check
            model_name: Model/method name to check for
            structures: Optional structures to check for

        Returns:
            List of patient IDs that need processing

        Example:
            >>> db = ResultsDatabase("/results")
            >>> all_patients = ["patient001", "patient002", "patient003"]
            >>> missing = db.get_missing_patients(all_patients, "sat_nano", ["myocardium"])
            >>> print(f"Need to process {len(missing)} patients: {missing}")
        """
        missing = []
        for patient_id in patient_list:
            if not self.result_exists(patient_id, model_name, structures):
                missing.append(patient_id)
        return missing

__all__ = [
    "ResultsDatabase",
]
