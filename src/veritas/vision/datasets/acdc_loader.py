"""ACDC (Automated Cardiac Diagnosis Challenge) dataset loader.

This module provides a loader for the ACDC cardiac MRI dataset, supporting:
- Loading single frames (ED/ES with annotations)
- Loading full 4D sequences (all cardiac phases)
- Training/testing split tracking
- Minimal raw loading + optional preprocessing
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..base import MedicalDatasetLoader, MedicalImage


class ACDCDatasetLoader(MedicalDatasetLoader):
    """
    Loader for ACDC cardiac MRI dataset.

    The ACDC dataset contains:
    - 150 cardiac cine MRI exams (100 training, 50 test)
    - 3D volumes at end-diastolic (ED) and end-systolic (ES) phases
    - Ground truth segmentations for LV, MYO, RV
    - Full 4D cardiac cycles (30 frames per patient)

    Dataset structure:
        data_root/
        ├── training/
        │   ├── patient001/
        │   │   ├── Info.cfg
        │   │   ├── patient001_4d.nii.gz          (full cardiac cycle)
        │   │   ├── patient001_frame01.nii.gz     (ED frame)
        │   │   ├── patient001_frame01_gt.nii.gz  (ED ground truth)
        │   │   ├── patient001_frame12.nii.gz     (ES frame)
        │   │   └── patient001_frame12_gt.nii.gz  (ES ground truth)
        │   └── ...
        └── testing/
            └── ...

    Usage:
        # Load single ED frame
        loader = ACDCDatasetLoader()
        ed_img = loader.load_image("patient001", frame="ED")

        # Load full 4D sequence
        seq_4d = loader.load_image("patient001", load_4d=True)
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = "training"
    ):
        """
        Initialize ACDC dataset loader.

        Args:
            data_root: Path to ACDC database directory. If not provided, uses
                DATASET_PATH, ACDC_DATA_ROOT, or ACDC_ROOT environment variables.
            split: Which data split to use ("training", "testing", or "all")
        """
        if data_root is None:
            data_root = (
                os.environ.get("DATASET_PATH")
                or os.environ.get("ACDC_DATA_ROOT")
                or os.environ.get("ACDC_ROOT")
            )

        if not data_root:
            raise ValueError(
                "ACDC data root not set. Provide data_root or set "
                "DATASET_PATH, ACDC_DATA_ROOT, or ACDC_ROOT."
            )
        self.data_root = Path(data_root).expanduser()
        self.split = split
        self._validate_data_root()

    # ========================================================================
    # Properties (required by MedicalDatasetLoader)
    # ========================================================================

    @property
    def name(self) -> str:
        """Dataset name."""
        return "ACDC"

    @property
    def supported_structures(self) -> List[str]:
        """List of anatomical structures in this dataset."""
        return ["LV", "MYO", "RV"]  # Left ventricle, myocardium, right ventricle

    @property
    def label_mapping(self) -> Dict[int, str]:
        """Ground truth label mapping."""
        return {
            0: "background",
            1: "RV",   # Right ventricle
            2: "MYO",  # Myocardium
            3: "LV"    # Left ventricle
        }

    @property
    def pathology_mapping(self) -> Dict[str, str]:
        """Pathology group name mapping."""
        return {
            "NOR": "Normal",
            "MINF": "Myocardial infarction",
            "DCM": "Dilated cardiomyopathy",
            "HCM": "Hypertrophic cardiomyopathy",
            "RV": "Abnormal right ventricle"
        }

    # ========================================================================
    # Listing and metadata methods
    # ========================================================================

    def list_available_images(self, split: Optional[str] = None) -> List[str]:
        """
        List available patient IDs.

        Args:
            split: Override default split ("training", "testing", "all")
                   If None, uses instance's split setting

        Returns:
            List of patient IDs (e.g., ['patient001', 'patient002', ...])
        """
        target_split = split or self.split

        patient_dirs = []

        if target_split in ["training", "all"]:
            training_dir = self.data_root / "training"
            if training_dir.exists():
                patient_dirs.extend([
                    d.name for d in training_dir.iterdir()
                    if d.is_dir() and d.name.startswith("patient")
                ])

        if target_split in ["testing", "all"]:
            testing_dir = self.data_root / "testing"
            if testing_dir.exists():
                patient_dirs.extend([
                    d.name for d in testing_dir.iterdir()
                    if d.is_dir() and d.name.startswith("patient")
                ])

        return sorted(patient_dirs)

    def get_patient_split(self, patient_id: str) -> str:
        """
        Determine which split a patient belongs to.

        Args:
            patient_id: Patient identifier (e.g., "patient001")

        Returns:
            "training" or "testing"

        Raises:
            ValueError: If patient not found in either split
        """
        if (self.data_root / "training" / patient_id).exists():
            return "training"
        elif (self.data_root / "testing" / patient_id).exists():
            return "testing"
        else:
            raise ValueError(
                f"Patient {patient_id} not found in training or testing directories"
            )

    def get_image_metadata(self, image_id: str) -> Dict[str, Any]:
        """
        Get patient metadata from Info.cfg.

        Args:
            image_id: Patient ID (e.g., "patient001")

        Returns:
            Dictionary with patient metadata including:
            - pathology_code: Disease group code
            - pathology_name: Full disease name
            - ed_frame: End-diastolic frame number
            - es_frame: End-systolic frame number
            - num_frames: Total frames in 4D sequence
            - height_cm, weight_kg: Patient demographics
            - split: "training" or "testing"
        """
        split = self.get_patient_split(image_id)
        patient_path = self.data_root / split / image_id

        info = self._parse_info_cfg(patient_path)

        return {
            'patient_id': image_id,
            'split': split,
            'pathology_code': info['pathology'],
            'pathology_name': self.pathology_mapping.get(
                info['pathology'], info['pathology']
            ),
            'ed_frame': info['ed_frame'],
            'es_frame': info['es_frame'],
            'num_frames': info['num_frames'],
            'height_cm': info['height'],
            'weight_kg': info['weight'],
        }

    def get_patient_metadata(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient metadata including ED/ES identifiers (for MCP tools).

        This method extends get_image_metadata() to include dataset-relative
        identifiers for ED and ES frames. The actual file paths are resolved
        internally by the dataset loader when needed.

        Args:
            patient_id: Patient ID (e.g., "patient001", "patient015")

        Returns:
            Dictionary with keys:
            - patient_id: Patient identifier
            - group: Group code (NOR, DCM, HCM, MINF, RV)
            - ed_frame: ED frame number (int)
            - es_frame: ES frame number (int)
            - num_frames: Total cardiac frames (int)
            - height: Patient height in cm (float)
            - weight: Patient weight in kg (float)
            - ed_identifier: Dataset-relative identifier for ED frame (str)
            - es_identifier: Dataset-relative identifier for ES frame (str)
            - observations: List of observation names ["ED", "ES"]
            - identifiers: Dict mapping observation names to full identifiers

        Example:
            >>> loader = ACDCDatasetLoader()
            >>> info = loader.get_patient_metadata("patient015")
            >>> print(info['es_frame'])  # 9
            >>> print(info['es_identifier'])   # acdc:patient015:ES
        """
        split = self.get_patient_split(patient_id)
        patient_dir = self.data_root / split / patient_id

        info = self._parse_info_cfg(patient_dir)

        # Build dataset-relative identifiers (dataset:patient:phase format)
        # These are resolved to full paths internally by the SAT MCP server
        ed_identifier = f"acdc:{patient_id}:ED"
        es_identifier = f"acdc:{patient_id}:ES"

        return {
            "patient_id": patient_id,
            "group": info['pathology'],
            "ed_frame": info['ed_frame'],
            "es_frame": info['es_frame'],
            "num_frames": info['num_frames'],
            "height": info['height'],
            "weight": info['weight'],
            "ed_identifier": ed_identifier,
            "es_identifier": es_identifier,
            # Add identifiers dict for dataset-agnostic access
            "observations": ["ED", "ES"],
            "identifiers": {
                "ED": ed_identifier,
                "ES": es_identifier,
            },
        }

    def list_patients_by_group(self, group: Optional[str] = None, metadata_filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        List all patients, optionally filtered by group.

        Args:
            group: Optional group filter:
                   - "NOR": Normal
                   - "DCM": Dilated cardiomyopathy
                   - "HCM": Hypertrophic cardiomyopathy
                   - "MINF": Myocardial infarction
                   - "RV": Abnormal right ventricle
                   - None: Return all patients

        Returns:
            List of dictionaries with patient_id, group, and num_frames

        Example:
            >>> loader = ACDCDatasetLoader()
            >>> dcm_patients = loader.list_patients_by_group(group="DCM")
            >>> print(f"Found {len(dcm_patients)} DCM patients")
        """
        all_patients = self.list_available_images()
        results = []

        for patient_id in all_patients:
            try:
                metadata = self.get_patient_metadata(patient_id)

                # Filter by group if specified
                if group is not None and metadata["group"] != group:
                    continue
                # Apply metadata filters (e.g. height, weight)
                if metadata_filters:
                    if not all(str(metadata.get(k)) == str(v) for k, v in metadata_filters.items()):
                        continue
                results.append({
                    "patient_id": patient_id,
                    "group": metadata["group"],
                    "num_frames": metadata["num_frames"]
                })
            except Exception:
                # Skip patients with invalid/missing metadata
                continue

        return results

    # ========================================================================
    # Main loading method
    # ========================================================================

    def load_image(
        self,
        image_id: str,
        frame: Union[str, int, None] = "ED",
        include_gt: bool = True,
        load_4d: bool = False
    ) -> MedicalImage:
        """
        Load ACDC image/volume.

        Args:
            image_id: Patient ID (e.g., "patient001")
            frame: Which frame to load:
                - "ED": End-diastolic frame (with GT)
                - "ES": End-systolic frame (with GT)
                - int: Specific frame number (1-based, no GT unless ED/ES)
                - None: Ignored if load_4d=True
            include_gt: Load ground truth segmentation (only for ED/ES)
            load_4d: Load full 4D sequence (all cardiac phases)
                     If True, returns 4D data: (time, slices, H, W)

        Returns:
            MedicalImage object with:
            - data: 3D (slices, H, W) or 4D (time, slices, H, W)
            - ground_truth: 3D (slices, H, W) if available
            - metadata: Comprehensive patient/image info

        Examples:
            # Load ED frame with ground truth (default)
            ed_img = loader.load_image("patient001")

            # Load ES frame
            es_img = loader.load_image("patient001", frame="ES")

            # Load specific frame without GT
            frame10 = loader.load_image("patient001", frame=10, include_gt=False)

            # Load full 4D cardiac cycle
            full_seq = loader.load_image("patient001", load_4d=True)
            # full_seq.data.shape == (30, 10, 256, 216)  # time, slices, H, W
        """
        split = self.get_patient_split(image_id)
        patient_path = self.data_root / split / image_id
        info_metadata = self._parse_info_cfg(patient_path)

        # Load 4D sequence if requested
        if load_4d:
            return self._load_4d_sequence(image_id, patient_path, info_metadata)

        # Determine frame number
        if frame == "ED":
            frame_num = info_metadata['ed_frame']
            phase_name = "ED"
        elif frame == "ES":
            frame_num = info_metadata['es_frame']
            phase_name = "ES"
        elif isinstance(frame, int):
            frame_num = frame
            phase_name = f"frame{frame:02d}"
        else:
            raise ValueError(f"Invalid frame specification: {frame}")

        # Load image volume
        image_path = patient_path / f"{image_id}_frame{frame_num:02d}.nii.gz"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_data, spacing_info = self._load_nifti_volume(image_path)

        # Load ground truth (only for ED/ES with include_gt=True)
        gt_data = None
        if include_gt and frame in ["ED", "ES"]:
            gt_path = patient_path / f"{image_id}_frame{frame_num:02d}_gt.nii.gz"
            if gt_path.exists():
                gt_data, _ = self._load_nifti_volume(gt_path)

        # Build comprehensive metadata
        metadata = {
            'patient_id': image_id,
            'frame_type': phase_name,
            'frame_number': frame_num,
            'is_annotated': frame in ["ED", "ES"],
            'split': split,
            'pathology': info_metadata['pathology'],
            'pathology_name': self.pathology_mapping.get(info_metadata['pathology']),
            'num_slices': image_data.shape[0],
            'num_frames_total': info_metadata['num_frames'],
            'ed_frame': info_metadata['ed_frame'],
            'es_frame': info_metadata['es_frame'],
            'origin': spacing_info['origin'],
            'direction': spacing_info['direction'],
        }

        return MedicalImage(
            image_id=f"{image_id}_{phase_name}",
            data=image_data,  # 3D: (slices, H, W)
            pixel_spacing=spacing_info['spacing'],
            modality="MRI",
            metadata=metadata,
            ground_truth=gt_data
        )

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _load_4d_sequence(
        self,
        image_id: str,
        patient_path: Path,
        info_metadata: Dict
    ) -> MedicalImage:
        """
        Load full 4D cardiac cycle.

        Args:
            image_id: Patient ID
            patient_path: Path to patient directory
            info_metadata: Parsed Info.cfg data

        Returns:
            MedicalImage with data.shape = (time, slices, H, W)
        """
        # Load full 4D sequence
        seq_path = patient_path / f"{image_id}_4d.nii.gz"
        if not seq_path.exists():
            raise FileNotFoundError(f"4D sequence not found: {seq_path}")

        seq_data, spacing_info = self._load_nifti_volume(seq_path)
        # seq_data shape: (time, slices, H, W)

        # Load ED ground truth (most common annotation)
        ed_frame = info_metadata['ed_frame']
        gt_path = patient_path / f"{image_id}_frame{ed_frame:02d}_gt.nii.gz"
        gt_data = None
        if gt_path.exists():
            gt_data, _ = self._load_nifti_volume(gt_path)

        metadata = {
            'patient_id': image_id,
            'frame_type': '4D_sequence',
            'is_4d': True,
            'num_frames': seq_data.shape[0],
            'num_slices': seq_data.shape[1],
            'split': self.get_patient_split(image_id),
            'pathology': info_metadata['pathology'],
            'pathology_name': self.pathology_mapping.get(info_metadata['pathology']),
            'ed_frame': info_metadata['ed_frame'],
            'es_frame': info_metadata['es_frame'],
            'origin': spacing_info['origin'],
            'direction': spacing_info['direction'],
        }

        return MedicalImage(
            image_id=f"{image_id}_4D",
            data=seq_data,  # 4D: (time, slices, H, W)
            pixel_spacing=spacing_info['spacing'],  # (x, y, z, t)
            modality="MRI",
            metadata=metadata,
            ground_truth=gt_data  # ED ground truth for reference
        )

    def _parse_info_cfg(self, patient_path: Path) -> Dict[str, Any]:
        """
        Parse Info.cfg file.

        Args:
            patient_path: Path to patient directory

        Returns:
            Dictionary with parsed configuration data
        """
        cfg_path = patient_path / "Info.cfg"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Info.cfg not found: {cfg_path}")

        metadata = {}

        with open(cfg_path) as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()

        return {
            'pathology': metadata.get('Group', 'UNKNOWN'),
            'ed_frame': int(metadata.get('ED', 1)),
            'es_frame': int(metadata.get('ES', 1)),
            'num_frames': int(metadata.get('NbFrame', 1)),
            'height': float(metadata.get('Height', 0.0)),
            'weight': float(metadata.get('Weight', 0.0)),
        }

    def _load_nifti_volume(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load NIfTI file using SimpleITK.

        Args:
            filepath: Path to .nii or .nii.gz file

        Returns:
            - array: numpy array (squeezed to remove singleton dimensions)
            - metadata: dict with origin, spacing, direction
        """
        import SimpleITK as sitk

        image = sitk.ReadImage(str(filepath))
        array = np.squeeze(sitk.GetArrayFromImage(image))

        metadata = {
            "origin": image.GetOrigin(),
            "spacing": image.GetSpacing(),
            "direction": image.GetDirection()
        }

        return array, metadata

    def _validate_data_root(self):
        """Validate that data_root contains expected ACDC structure."""
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")

        # Check for training or testing directories
        has_training = (self.data_root / "training").exists()
        has_testing = (self.data_root / "testing").exists()

        if not (has_training or has_testing):
            raise ValueError(
                f"Expected 'training' or 'testing' directories in {self.data_root}"
            )

    # ========================================================================
    # Optional preprocessing methods
    # ========================================================================

    def preprocess_normalize_intensity(
        self,
        image: np.ndarray,
        method: str = "minmax"
    ) -> np.ndarray:
        """
        Normalize image intensity (optional preprocessing).

        Args:
            image: Input image array
            method: Normalization method:
                - "minmax": Scale to [0, 1]
                - "zscore": Zero mean, unit variance
                - "percentile": Clip to 1st-99th percentile, then scale

        Returns:
            Normalized image
        """
        if method == "minmax":
            min_val, max_val = image.min(), image.max()
            return (image - min_val) / (max_val - min_val + 1e-8)
        elif method == "zscore":
            mean, std = image.mean(), image.std()
            return (image - mean) / (std + 1e-8)
        elif method == "percentile":
            p1, p99 = np.percentile(image, [1, 99])
            image_clipped = np.clip(image, p1, p99)
            return (image_clipped - p1) / (p99 - p1 + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def preprocess_resample_spacing(
        self,
        image: np.ndarray,
        current_spacing: Tuple[float, ...],
        target_spacing: Tuple[float, ...],
        order: int = 3
    ) -> np.ndarray:
        """
        Resample image to target spacing (optional preprocessing).

        Uses scipy.ndimage.zoom for resampling.

        Args:
            image: Input image array
            current_spacing: Current pixel spacing (mm/pixel)
            target_spacing: Target pixel spacing (mm/pixel)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)

        Returns:
            Resampled image
        """
        from scipy import ndimage

        # Calculate zoom factors
        zoom_factors = np.array(current_spacing[:image.ndim]) / np.array(target_spacing[:image.ndim])

        # Resample
        resampled = ndimage.zoom(image, zoom_factors, order=order)

        return resampled
