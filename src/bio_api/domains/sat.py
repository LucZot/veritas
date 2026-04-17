"""SAT (Segment Anything for Medical Images) Segmentation API."""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from bio_api.base import BaseAPI


class SATAPI(BaseAPI):
    """SAT (Segment Anything for Medical Images) Segmentation API.

    Quick Start:
        from bio_api import registry

        db_path = '/path/to/results_db'

        # List patients
        patients = registry.sat.list_patients(db_path)

        # Load structure masks (returns list - one per source image)
        masks = registry.sat.load_structure_mask(db_path, 'patient001', 'liver')

        # Calculate volumes and mass
        for m in masks:
            vol = registry.sat.calculate_volume(m['mask'], m['spacing'])
            print(f"{m['source_image']}: {vol:.1f} mL")
            mass = registry.sat.calculate_mass(m['mask'], m['spacing'])
            print(f"{m['source_image']}: {mass:.1f} g")

    Methods:
        # Data Access
        list_results(db_path) - Browse all results with metadata
        list_patients(db_path) - List patient IDs
        load_structure_mask(db_path, patient_id, structure) - Load masks (returns list)
        get_patient_metadata(patient_id, dataset_path) - Get ED/ES frames and clinical info

        # Basic Metrics
        calculate_volume(mask, spacing) - Calculate volume in mL
        calculate_mass(mask, spacing, density) - Calculate mass in grams

        # Advanced Shape Metrics (domain-specific algorithms)
        calculate_surface_area(mask, spacing) - Surface area via marching cubes (mm²)
        calculate_sphericity_index(mask, spacing) - Cardiac remodeling metric (short/long axis)
        calculate_wall_thickness(inner_mask, outer_mask, spacing) - Myocardial wall analysis

        # Convenience Functions
        calculate_ejection_fraction(ed_mask, es_mask, spacing) - Direct LVEF calculation
    """

    @property
    def api_name(self) -> str:
        return "sat"

    def list_patients(self, results_db_path: str) -> List[str]:
        """List all unique patient IDs in the database.

        Args:
            results_db_path: Path to results_database directory

        Returns:
            Sorted list of patient IDs
        """
        manifest_path = Path(results_db_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        patient_ids = sorted(set(r['patient_id'] for r in manifest['results']))
        return patient_ids

    def list_results(
        self,
        results_db_path: str,
        patient_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all segmentation results in the database.

        Args:
            results_db_path: Path to results_database directory
            patient_id: Optional - filter to specific patient

        Returns:
            List of result entries with: patient_id, source_image, structures, etc.

        Example:
            >>> results = registry.sat.list_results(db_path)
            >>> for r in results:
            ...     print(f"{r['patient_id']}: {r['source_image']} -> {r['structures']}")
        """
        manifest_path = Path(results_db_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        results = manifest.get('results', [])

        if patient_id:
            results = [r for r in results if r['patient_id'] == patient_id]

        return results

    def load_structure_mask(
        self,
        results_db_path: str,
        patient_id: str,
        structure_name: str,
        source_image_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load all masks for a structure across all results for a patient.

        Args:
            results_db_path: Path to results_database directory
            patient_id: Patient ID (e.g., 'patient001'). Do not pass dataset identifiers
                        like "acdc:patient001:ED"; extract the patient ID first.
            structure_name: Structure name (e.g., 'liver', 'left heart ventricle')
            source_image_contains: Optional substring filter for source_image.
                                   Use to load only ED/ES frames (e.g., "frame01").

        Returns:
            List of dictionaries, each with: mask, spacing, source_image, patient_id

        Example:
            >>> # Load all masks for a structure
            >>> masks = registry.sat.load_structure_mask(db_path, 'patient001', 'liver')
            >>> print(f"Found {len(masks)} masks")
            >>> for m in masks:
            ...     vol = registry.sat.calculate_volume(m['mask'], m['spacing'])
            ...     print(f"{m['source_image']}: {vol:.1f} mL")
        """
        import nibabel as nib

        # Load manifest
        manifest_path = Path(results_db_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Filter results for this patient
        patient_results = [
            r for r in manifest['results']
            if r['patient_id'] == patient_id
        ]

        if not patient_results:
            raise ValueError(f"No results found for patient '{patient_id}'")

        # Smart extension handling - remove common extensions FIRST (before normalization)
        # This allows alias matching to work even if agent includes extensions
        clean_structure_name = structure_name.replace('.nii.gz', '').replace('.nii', '').strip()

        # Normalize structure aliases AFTER stripping extensions
        alias_map = {
            "left ventricle": "left heart ventricle",
            "lv": "left heart ventricle",
            "right ventricle": "right heart ventricle",
            "rv": "right heart ventricle",
            "myo": "myocardium",
        }
        normalized = clean_structure_name.lower()
        clean_structure_name = alias_map.get(normalized, clean_structure_name)

        # Load mask for each result
        loaded_masks = []
        for result in patient_results:
            if source_image_contains:
                source_image = result.get("source_image", "")
                if source_image_contains not in source_image:
                    continue

            structures_dir = result.get('structures_dir')
            if not structures_dir:
                continue

            structures_dir = Path(structures_dir)
            if not structures_dir.exists():
                continue

            structure_mask_path = structures_dir / f"{clean_structure_name}.nii.gz"
            if not structure_mask_path.exists():
                continue

            # Load mask efficiently - segmentation masks are label images, not float
            # Convert to uint8 to reduce memory by 8x (critical for large datasets)
            # get_fdata() returns float by default; convert immediately to save memory
            nii_img = nib.load(structure_mask_path)
            mask = nii_img.get_fdata().astype(np.uint8)
            spacing = nii_img.header.get_zooms()

            loaded_masks.append({
                'patient_id': patient_id,
                'mask': mask,
                'spacing': spacing,
                'source_image': result.get('source_image', ''),
                'structure_name': clean_structure_name,
                'mask_path': str(structure_mask_path),
            })

        if not loaded_masks:
            # List available structures from first result
            first_dir = patient_results[0].get('structures_dir')
            if first_dir and Path(first_dir).exists():
                available = [f.stem for f in Path(first_dir).glob("*.nii.gz")]
                raise FileNotFoundError(
                    f"Structure '{clean_structure_name}' not found. Available: {available}"
                )
            raise FileNotFoundError(f"No masks found for '{clean_structure_name}'")

        return loaded_masks

    def calculate_volume(self, mask, spacing: tuple, label_id: int = 1) -> float:
        """Calculate volume of a structure in a segmentation mask.

        Args:
            mask: Segmentation mask (numpy array)
            spacing: Voxel spacing in mm as tuple (e.g., (1.4, 1.4, 10.0))
            label_id: Label ID to count (default: 1 for binary masks)

        Returns:
            Volume in milliliters (mL)

        Example:
            >>> masks = registry.sat.load_structure_mask(db_path, 'patient001', 'liver')
            >>> volume = registry.sat.calculate_volume(masks[0]['mask'], masks[0]['spacing'])
            >>> print(f"Volume: {volume:.2f} mL")
        """
        import numpy as np

        # Count voxels for this label
        label_voxels = np.sum(mask == label_id)

        # Calculate voxel volume in mm^3
        voxel_volume_mm3 = np.prod(spacing)

        # Calculate total volume in mm^3, then convert to mL
        volume_mm3 = label_voxels * voxel_volume_mm3
        volume_ml = volume_mm3 / 1000.0

        return volume_ml

    def calculate_mass(
        self,
        mask,
        spacing: tuple,
        label_id: int = 1,
        density_g_ml: float = 1.05
    ) -> float:
        """Calculate mass of a structure in a segmentation mask.

        Uses volume (mL) * density (g/mL). Default density is myocardium (1.05).

        Args:
            mask: Segmentation mask (numpy array)
            spacing: Voxel spacing in mm as tuple (e.g., (1.4, 1.4, 10.0))
            label_id: Label ID to count (default: 1 for binary masks)
            density_g_ml: Tissue density in g/mL (default: 1.05)

        Returns:
            Mass in grams (g)

        Example:
            >>> masks = registry.sat.load_structure_mask(db_path, 'patient001', 'myocardium')
            >>> mass = registry.sat.calculate_mass(masks[0]['mask'], masks[0]['spacing'])
            >>> print(f"Mass: {mass:.2f} g")
        """
        volume_ml = self.calculate_volume(mask, spacing, label_id=label_id)
        return float(volume_ml * density_g_ml)

    def get_unique_labels(self, mask) -> List[int]:
        """Get unique label IDs present in a segmentation mask.

        Args:
            mask: Segmentation mask (numpy array)

        Returns:
            Sorted list of unique label IDs

        Example:
            >>> masks = registry.sat.load_structure_mask(db_path, 'patient001', 'liver')
            >>> labels = registry.sat.get_unique_labels(masks[0]['mask'])
            >>> print(f"Labels: {labels}")  # [0, 1] for binary mask
        """
        import numpy as np
        return sorted(int(x) for x in np.unique(mask).tolist())

    def list_patient_source_images(
        self,
        results_db_path: str,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """List all available source images (scans/frames) for a patient.

        Useful for dataset-agnostic workflows where you need to discover
        what observations/timepoints are available for each patient.

        Args:
            results_db_path: Path to results_database directory
            patient_id: Patient ID (e.g., 'patient001')

        Returns:
            List of dicts with 'source_image' paths, one per scan/frame

        Example:
            >>> # Discover all available scans for a patient
            >>> sources = registry.sat.list_patient_source_images(db_path, 'patient001')
            >>> for src in sources:
            ...     print(src['source_image'])
            # /path/to/patient001_frame01.nii.gz
            # /path/to/patient001_frame12.nii.gz
            # ... (all frames)

            >>> # Load masks for all frames
            >>> for src in sources:
            ...     frame_id = Path(src['source_image']).stem  # 'patient001_frame01'
            ...     masks = sat.load_structure_mask(db_path, patient_id, 'liver',
            ...                                      source_image_contains=frame_id)
        """
        # Load manifest
        manifest_path = Path(results_db_path) / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Filter results for this patient
        patient_results = [
            {'source_image': r['source_image'], 'result_index': i}
            for i, r in enumerate(manifest['results'])
            if r['patient_id'] == patient_id
        ]

        if not patient_results:
            raise ValueError(f"No results found for patient '{patient_id}'")

        return patient_results

    def get_observation_identifiers(
        self,
        patient_id: str,
        dataset_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Get observation → identifier mapping for a patient.

        Returns a dict mapping observation names (e.g., 'ED', 'ES', or 'frame01', 'frame02')
        to dataset identifiers that can be used with source_image_contains.

        This provides a dataset-agnostic way to access observations without
        knowing the specific dataset format.

        Args:
            patient_id: Patient ID (e.g., 'patient001')
            dataset_path: Path to dataset root directory (optional)

        Returns:
            Dict mapping observation names to identifiers:
            - For ACDC: {'ED': 'frame01', 'ES': 'frame12', ...}
            - For other datasets: observation → identifier mapping from metadata

        Example:
            >>> # Get observation mappings
            >>> obs_map = registry.sat.get_observation_identifiers('patient001')
            >>> print(obs_map)
            # {'ED': 'frame01', 'ES': 'frame12'}

            >>> # Load ED mask using the mapping
            >>> ed_id = obs_map['ED']
            >>> ed_masks = sat.load_structure_mask(db_path, 'patient001', 'left ventricle',
            ...                                     source_image_contains=ed_id)
        """
        import os

        # Get full metadata
        metadata = self.get_patient_metadata(patient_id, dataset_path)

        # Extract observation identifiers
        observation_map = {}

        # Check if metadata has 'identifiers' dict (manifest format)
        if 'identifiers' in metadata:
            # Manifest format: parse identifiers like "acdc:patient001:ED"
            for obs_name, identifier in metadata['identifiers'].items():
                # Extract the actual identifier from dataset:patient:obs format
                parts = identifier.split(':')
                if len(parts) == 3:
                    # For ACDC: convert observation to frame number
                    if obs_name == 'ED' and 'ed_frame' in metadata:
                        observation_map[obs_name] = f"frame{metadata['ed_frame']:02d}"
                    elif obs_name == 'ES' and 'es_frame' in metadata:
                        observation_map[obs_name] = f"frame{metadata['es_frame']:02d}"
                    else:
                        observation_map[obs_name] = obs_name
                else:
                    observation_map[obs_name] = identifier

        # Fall back to frame-based mapping for ACDC
        elif 'ed_frame' in metadata or 'es_frame' in metadata:
            if 'ed_frame' in metadata:
                observation_map['ED'] = f"frame{metadata['ed_frame']:02d}"
            if 'es_frame' in metadata:
                observation_map['ES'] = f"frame{metadata['es_frame']:02d}"

            # Add all frame numbers if num_frames is available
            if 'num_frames' in metadata:
                for frame_num in range(1, metadata['num_frames'] + 1):
                    observation_map[f"frame{frame_num:02d}"] = f"frame{frame_num:02d}"

        return observation_map

    def get_patient_metadata(
        self,
        patient_id: str,
        dataset_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get patient metadata including ED/ES frames and clinical information.

        For ACDC dataset: Reads from Info.cfg file
        For manifest datasets: Reads from dataset manifest if present
        For custom datasets: Reads from patient_metadata.json (if exists)

        Args:
            patient_id: Patient ID (e.g., 'patient001')
            dataset_path: Path to dataset root directory (optional)
                         If not provided, tries DATASET_PATH, ACDC_DATA_ROOT,
                         or ACDC_ROOT environment variables
                         Accepts either dataset root or its /database subdirectory

        Returns:
            Dictionary with patient metadata:
                - ed_frame (int): End-diastole frame number
                - es_frame (int): End-systole frame number
                - group (str): Disease group (e.g., 'DCM', 'NOR', 'GradeIV')
                - num_frames (int): Total frames in cardiac cycle
                - height (float): Patient height in cm
                - weight (float): Patient weight in kg

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If dataset_path cannot be determined

        Example:
            >>> # Get metadata for ACDC patient
            >>> metadata = registry.sat.get_patient_metadata('patient001')
            >>> print(f"ED frame: {metadata['ed_frame']}")  # 1
            >>> print(f"ES frame: {metadata['es_frame']}")  # 12 (varies per patient)
            >>> print(f"Group: {metadata['group']}")  # DCM

            >>> # Use metadata to find correct frames
            >>> lv_masks = registry.sat.load_structure_mask(db_path, 'patient001', 'left ventricle')
            >>> ed_frame = metadata['ed_frame']
            >>> es_frame = metadata['es_frame']
            >>> ed_mask = [m for m in lv_masks if f'frame{ed_frame:02d}' in m['source_image']][0]
            >>> es_mask = [m for m in lv_masks if f'frame{es_frame:02d}' in m['source_image']][0]
        """
        import os
        import re

        # Determine dataset path
        if dataset_path is None:
            dataset_path = (
                os.environ.get('DATASET_PATH')
                or os.environ.get('ACDC_DATA_ROOT')
                or os.environ.get('ACDC_ROOT')
            )
            if not dataset_path:
                raise ValueError(
                    f"Cannot find dataset path for patient '{patient_id}'. "
                    "Please provide dataset_path or set DATASET_PATH, "
                    "ACDC_DATA_ROOT, or ACDC_ROOT."
                )

        dataset_root = Path(dataset_path)
        if dataset_root.name == "database":
            dataset_root = dataset_root.parent

        # Try manifest dataset first
        try:
            from veritas.vision.datasets.manifest_loader import ManifestDatasetLoader
            manifest_loader = ManifestDatasetLoader(dataset_root)
            return manifest_loader.get_patient_metadata(patient_id)
        except FileNotFoundError:
            pass

        # Try ACDC format (Info.cfg) - check both training and testing
        patient_dir = None
        for split in ["training", "testing"]:
            candidate_dir = dataset_root / "database" / split / patient_id
            if (candidate_dir / "Info.cfg").exists() or (
                candidate_dir / "patient_metadata.json"
            ).exists():
                patient_dir = candidate_dir
                break

        if patient_dir is None:
            # Patient not found in either split
            raise FileNotFoundError(
                f"Patient '{patient_id}' not found in training or testing directories. "
                f"Searched: {dataset_root / 'database' / 'training' / patient_id}, "
                f"{dataset_root / 'database' / 'testing' / patient_id}"
            )

        info_cfg_path = patient_dir / "Info.cfg"

        if info_cfg_path.exists():
            # Parse ACDC Info.cfg
            metadata_raw = {}
            with open(info_cfg_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        metadata_raw[key.strip()] = value.strip()

            return {
                'ed_frame': int(metadata_raw.get('ED', 1)),
                'es_frame': int(metadata_raw.get('ES', 1)),
                'group': metadata_raw.get('Group', 'UNKNOWN'),
                'num_frames': int(metadata_raw.get('NbFrame', 1)),
                'height': float(metadata_raw.get('Height', 0.0)),
                'weight': float(metadata_raw.get('Weight', 0.0)),
            }

        # Try custom metadata format (patient_metadata.json)
        metadata_json_path = patient_dir / "patient_metadata.json"
        if metadata_json_path.exists():
            with open(metadata_json_path, 'r') as f:
                return json.load(f)

        # Metadata file not found
        raise FileNotFoundError(
            f"Metadata file not found for patient '{patient_id}'. "
            f"Tried: {info_cfg_path}, {metadata_json_path}"
        )

    # ==================== ADVANCED SHAPE METRICS ====================

    def calculate_surface_area(
        self,
        mask,
        spacing: tuple,
        label_id: int = 1
    ) -> float:
        """Calculate surface area of a 3D structure using marching cubes algorithm.

        Uses the marching cubes algorithm to generate a mesh surface, then computes
        the total surface area. Useful for surface-to-volume ratio analysis and
        remodeling studies.

        Args:
            mask: 3D segmentation mask (numpy array)
            spacing: Voxel spacing in mm as tuple (e.g., (1.4, 1.4, 10.0))
            label_id: Label ID to analyze (default: 1 for binary masks)

        Returns:
            Surface area in square millimeters (mm²)

        Example:
            >>> # Calculate LV endocardial surface area
            >>> masks = registry.sat.load_structure_mask(db_path, 'patient001', 'left ventricle')
            >>> surface_area = registry.sat.calculate_surface_area(masks[0]['mask'], masks[0]['spacing'])
            >>> print(f"Surface area: {surface_area:.1f} mm²")
            >>>
            >>> # Surface-to-volume ratio (shape complexity indicator)
            >>> volume = registry.sat.calculate_volume(masks[0]['mask'], masks[0]['spacing'])
            >>> sv_ratio = surface_area / (volume * 1000)  # volume in mL -> mm³
            >>> print(f"Surface/Volume ratio: {sv_ratio:.4f} mm⁻¹")

        Note:
            - Requires scikit-image library
            - Accuracy depends on voxel resolution
            - Use isotropic spacing for best results (or resample first)
        """
        import numpy as np
        from skimage import measure

        # Create binary mask for this label
        binary_mask = (mask == label_id).astype(np.uint8)

        # Run marching cubes to extract surface mesh
        # Level 0.5 is standard for binary masks
        verts, faces, normals, values = measure.marching_cubes(
            binary_mask,
            level=0.5,
            spacing=spacing
        )

        # Calculate surface area from mesh triangles
        # Each face is a triangle with 3 vertices
        surface_area = 0.0
        for face in faces:
            # Get triangle vertices
            v0 = verts[face[0]]
            v1 = verts[face[1]]
            v2 = verts[face[2]]

            # Calculate triangle area using cross product
            # Area = 0.5 * ||(v1-v0) × (v2-v0)||
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            triangle_area = 0.5 * np.linalg.norm(cross)
            surface_area += triangle_area

        return float(surface_area)

    def calculate_sphericity_index(
        self,
        mask,
        spacing: tuple,
        label_id: int = 1
    ) -> float:
        """Calculate sphericity index (short axis / long axis ratio).

        Sphericity index is a measure of how spherical a structure is. For cardiac
        imaging, it quantifies ventricular remodeling. Normal hearts are more elongated
        (sphericity < 0.6), while dilated cardiomyopathy shows increased sphericity
        (more spherical shape).

        Research shows sphericity index is an early marker of cardiomyopathy and
        correlates with adverse outcomes (PMID: 37124279).

        Args:
            mask: 3D segmentation mask (numpy array)
            spacing: Voxel spacing in mm as tuple (e.g., (1.4, 1.4, 10.0))
            label_id: Label ID to analyze (default: 1 for binary masks)

        Returns:
            Sphericity index (unitless ratio, typically 0.4-1.0)
            - 1.0 = perfect sphere
            - < 0.6 = elongated (normal LV)
            - > 0.7 = spherical (dilated/remodeled)

        Example:
            >>> # Compare LV sphericity between DCM and normal
            >>> lv_masks = registry.sat.load_structure_mask(db_path, 'patient001', 'left ventricle')
            >>> ed_mask = [m for m in lv_masks if 'frame01' in m['source_image']][0]
            >>> sphericity = registry.sat.calculate_sphericity_index(ed_mask['mask'], ed_mask['spacing'])
            >>> print(f"Sphericity index: {sphericity:.3f}")
            >>> if sphericity > 0.7:
            >>>     print("Spherical remodeling detected")

        Clinical Interpretation:
            - Normal LV: 0.45-0.60 (elongated, bullet-shaped)
            - Mild remodeling: 0.60-0.70
            - Severe remodeling (DCM): > 0.70 (spherical)

        Note:
            - Uses principal component analysis to find major axes
            - Short axis = largest diameter perpendicular to long axis
            - Long axis = largest extent of the structure
        """
        import numpy as np
        from scipy.ndimage import center_of_mass

        # Get coordinates of all voxels in this structure
        binary_mask = (mask == label_id)
        coords = np.argwhere(binary_mask)

        if len(coords) < 10:
            raise ValueError(f"Insufficient voxels ({len(coords)}) for sphericity calculation")

        # Convert voxel coordinates to physical coordinates (mm)
        physical_coords = coords * np.array(spacing)

        # Center the coordinates
        centroid = np.mean(physical_coords, axis=0)
        centered_coords = physical_coords - centroid

        # Principal Component Analysis - find principal axes
        # Covariance matrix of coordinates
        cov_matrix = np.cov(centered_coords.T)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue magnitude (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]

        # Principal axis lengths are proportional to sqrt(eigenvalues)
        # Long axis = largest principal component
        # Short axis = second largest principal component
        long_axis = 2 * np.sqrt(eigenvalues[0])  # 2σ captures ~95% of distribution
        short_axis = 2 * np.sqrt(eigenvalues[1])

        # Sphericity index
        sphericity = short_axis / long_axis

        return float(sphericity)

    def calculate_wall_thickness(
        self,
        inner_mask,
        outer_mask,
        spacing: tuple,
        label_id: int = 1
    ) -> Dict[str, float]:
        """Calculate myocardial wall thickness from nested masks.

        Computes wall thickness by calculating the distance from the inner surface
        (e.g., LV endocardium) to the outer surface (e.g., LV epicardium). This is
        critical for detecting hypertrophy, regional wall thinning, and remodeling.

        Args:
            inner_mask: Inner boundary mask (e.g., LV cavity)
            outer_mask: Outer boundary mask (e.g., LV + myocardium)
            spacing: Voxel spacing in mm as tuple (e.g., (1.4, 1.4, 10.0))
            label_id: Label ID to analyze (default: 1 for binary masks)

        Returns:
            Dictionary with wall thickness statistics:
                - mean: Mean wall thickness (mm)
                - std: Standard deviation (mm)
                - min: Minimum thickness (mm)
                - max: Maximum thickness (mm)
                - median: Median thickness (mm)

        Example:
            >>> # Calculate LV wall thickness for hypertrophy assessment
            >>> lv_masks = registry.sat.load_structure_mask(db_path, 'patient001', 'left heart ventricle')
            >>> myo_masks = registry.sat.load_structure_mask(db_path, 'patient001', 'myocardium')
            >>>
            >>> # Get ED frame masks
            >>> lv_ed = [m for m in lv_masks if 'frame01' in m['source_image']][0]
            >>> myo_ed = [m for m in myo_masks if 'frame01' in m['source_image']][0]
            >>>
            >>> # Outer mask = LV + myocardium (combine masks)
            >>> import numpy as np
            >>> outer = np.logical_or(lv_ed['mask'], myo_ed['mask']).astype(np.uint8)
            >>>
            >>> thickness = registry.sat.calculate_wall_thickness(
            >>>     lv_ed['mask'], outer, lv_ed['spacing']
            >>> )
            >>> print(f"Mean wall thickness: {thickness['mean']:.1f} mm")
            >>> print(f"Range: {thickness['min']:.1f} - {thickness['max']:.1f} mm")
            >>>
            >>> # Clinical interpretation
            >>> if thickness['mean'] > 12:
            >>>     print("Hypertrophy detected (normal: 6-11 mm)")

        Clinical Reference Values:
            - Normal LV wall: 6-11 mm
            - Mild hypertrophy: 12-14 mm
            - Severe hypertrophy (HCM): > 15 mm
            - Wall thinning (MINF): < 6 mm

        Note:
            - Uses Euclidean distance transform
            - Assumes inner_mask is completely inside outer_mask
            - Wall = outer_mask XOR inner_mask (myocardium region)
        """
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        # Create binary masks
        inner_binary = (inner_mask == label_id).astype(bool)
        outer_binary = (outer_mask == label_id).astype(bool)

        # Validate that inner is contained in outer
        if not np.all(inner_binary <= outer_binary):
            raise ValueError("Inner mask must be completely contained within outer mask")

        # Calculate wall region (myocardium = outer XOR inner)
        wall_region = np.logical_xor(outer_binary, inner_binary)

        if not np.any(wall_region):
            raise ValueError("No wall region found (outer and inner masks are identical)")

        # Distance transform from inner surface
        # For each wall voxel, compute distance to nearest inner surface voxel
        distance_from_inner = distance_transform_edt(~inner_binary, sampling=spacing)

        # Distance transform from outer surface
        distance_from_outer = distance_transform_edt(~outer_binary, sampling=spacing)

        # Wall thickness at each voxel = sum of distances
        # (distance to inner + distance to outer surface)
        thickness_map = distance_from_inner + distance_from_outer

        # Extract thickness values only in the wall region
        wall_thickness_values = thickness_map[wall_region]

        # Compute statistics
        return {
            'mean': float(np.mean(wall_thickness_values)),
            'std': float(np.std(wall_thickness_values)),
            'min': float(np.min(wall_thickness_values)),
            'max': float(np.max(wall_thickness_values)),
            'median': float(np.median(wall_thickness_values)),
        }

    def calculate_ejection_fraction(
        self,
        ed_mask,
        es_mask,
        spacing: tuple,
        label_id: int = 1
    ) -> float:
        """Calculate ejection fraction (EF) from end-diastole and end-systole masks.

        Convenience function that computes EF = ((EDV - ESV) / EDV) × 100.
        This is the primary metric for assessing cardiac pump function.

        Args:
            ed_mask: End-diastole segmentation mask
            es_mask: End-systole segmentation mask
            spacing: Voxel spacing in mm as tuple (e.g., (1.4, 1.4, 10.0))
            label_id: Label ID to analyze (default: 1 for binary masks)

        Returns:
            Ejection fraction as percentage (0-100)

        Example:
            >>> # Calculate LVEF for a patient
            >>> lv_masks = registry.sat.load_structure_mask(db_path, 'patient001', 'left heart ventricle')
            >>>
            >>> # Get ED and ES frames from metadata
            >>> metadata = registry.sat.get_patient_metadata('patient001')
            >>> ed_frame = metadata['ed_frame']
            >>> es_frame = metadata['es_frame']
            >>>
            >>> ed_mask = [m for m in lv_masks if f'frame{ed_frame:02d}' in m['source_image']][0]
            >>> es_mask = [m for m in lv_masks if f'frame{es_frame:02d}' in m['source_image']][0]
            >>>
            >>> lvef = registry.sat.calculate_ejection_fraction(
            >>>     ed_mask['mask'], es_mask['mask'], ed_mask['spacing']
            >>> )
            >>> print(f"LVEF: {lvef:.1f}%")

        Clinical Reference Values:
            - Normal LVEF: ≥ 55%
            - Mildly reduced: 45-54%
            - Moderately reduced: 30-44%
            - Severely reduced (DCM): < 30%

        Note:
            - EDV = End-diastolic volume (largest volume)
            - ESV = End-systolic volume (smallest volume)
            - Both masks should use the same spacing
        """
        # Calculate volumes
        edv = self.calculate_volume(ed_mask, spacing, label_id=label_id)
        esv = self.calculate_volume(es_mask, spacing, label_id=label_id)

        # Validate EDV > ESV (should always be true physiologically)
        if edv < esv:
            raise ValueError(
                f"End-diastolic volume ({edv:.1f} mL) is less than "
                f"end-systolic volume ({esv:.1f} mL). Check frame labels."
            )

        # Calculate EF
        ef = ((edv - esv) / edv) * 100.0

        return float(ef)
