"""Utility functions for vision processing."""

import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List, Any


def create_segmentation_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0)
) -> Image.Image:
    """Create visualization with mask overlay.

    Args:
        image: Source PIL Image (RGB)
        mask: Binary mask (H, W), values 0 or 1
        alpha: Overlay transparency (0=invisible, 1=opaque)
        color: RGB color for mask overlay (default: red)

    Returns:
        PIL Image with overlay
    """
    img_array = np.array(image.convert("RGB"))

    # Create colored overlay
    overlay = np.zeros_like(img_array)
    overlay[mask > 0] = color

    # Blend
    blended = (img_array * (1 - alpha) + overlay * alpha).astype(np.uint8)

    return Image.fromarray(blended)


def compute_dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice similarity coefficient between prediction and ground truth.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        Dice score (0-1, higher is better)
    """
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


def normalize_medical_image(data: np.ndarray) -> np.ndarray:
    """
    Normalize medical image data to 0-255 range.

    Args:
        data: Raw image data (any range)

    Returns:
        Normalized array (uint8, 0-255)
    """
    normalized = ((data - data.min()) /
                 (data.max() - data.min()) * 255).astype(np.uint8)
    return normalized


def save_mask(mask: np.ndarray, path: str) -> None:
    """
    Save binary mask as PNG.

    Args:
        mask: Binary mask (H, W)
        path: Output file path
    """
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(path)


# ============================================================================
# ACDC-specific cardiac measurement utilities
# ============================================================================

def calculate_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_id: Optional[int] = None
) -> float:
    """
    Calculate Dice coefficient for segmentation evaluation.

    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        class_id: Specific class to evaluate (None = all non-zero pixels)

    Returns:
        Dice score in range [0, 1], where 1 = perfect overlap

    Example:
        >>> dice_lv = calculate_dice_score(pred, gt, class_id=3)  # LV only
        >>> dice_all = calculate_dice_score(pred, gt)  # All structures
    """
    if class_id is not None:
        pred = (pred_mask == class_id).astype(float)
        gt = (gt_mask == class_id).astype(float)
    else:
        pred = (pred_mask > 0).astype(float)
        gt = (gt_mask > 0).astype(float)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


def calculate_cardiac_volume(
    mask_3d: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
    class_id: int = 3
) -> Dict[str, float]:
    """
    Calculate volume from 3D segmentation mask.

    Args:
        mask_3d: 3D segmentation mask with shape (slices, H, W)
        pixel_spacing: Physical spacing (x_mm, y_mm, z_mm)
        class_id: Which structure to measure:
                  1 = RV (right ventricle)
                  2 = MYO (myocardium)
                  3 = LV (left ventricle, default)

    Returns:
        Dictionary containing:
        - volume_ml: Volume in milliliters
        - volume_mm3: Volume in cubic millimeters
        - num_slices_with_label: Number of slices containing the structure
        - areas_per_slice_mm2: List of cross-sectional areas per slice

    Example:
        >>> ed_vol = calculate_cardiac_volume(ed_gt, spacing, class_id=3)
        >>> print(f"LV volume: {ed_vol['volume_ml']:.2f} ml")
    """
    # Extract binary mask for specified class
    binary_mask = (mask_3d == class_id).astype(float)

    # Calculate voxel volume in mm³
    voxel_vol_mm3 = pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]

    # Total volume
    num_voxels = binary_mask.sum()
    volume_mm3 = num_voxels * voxel_vol_mm3
    volume_ml = volume_mm3 / 1000.0  # 1 ml = 1000 mm³

    # Per-slice areas
    areas_per_slice = []
    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]

    for slice_idx in range(mask_3d.shape[0]):
        slice_mask = binary_mask[slice_idx]
        area_mm2 = slice_mask.sum() * pixel_area_mm2
        areas_per_slice.append(area_mm2)

    num_slices_with_label = sum(1 for area in areas_per_slice if area > 0)

    return {
        'volume_ml': volume_ml,
        'volume_mm3': volume_mm3,
        'num_slices_with_label': num_slices_with_label,
        'areas_per_slice_mm2': areas_per_slice
    }


def calculate_ejection_fraction(
    ed_volume_ml: float,
    es_volume_ml: float
) -> float:
    """
    Calculate left ventricular ejection fraction.

    Ejection Fraction (EF) = (EDV - ESV) / EDV × 100%

    EF is a key cardiac function metric:
    - Normal: 55-70%
    - Mildly reduced: 45-54%
    - Moderately reduced: 30-44%
    - Severely reduced: <30%

    Args:
        ed_volume_ml: End-diastolic volume (ml)
        es_volume_ml: End-systolic volume (ml)

    Returns:
        Ejection fraction as percentage (0-100)

    Example:
        >>> ef = calculate_ejection_fraction(120.0, 50.0)
        >>> print(f"EF: {ef:.1f}%")  # EF: 58.3%
    """
    if ed_volume_ml == 0:
        return 0.0

    ef = ((ed_volume_ml - es_volume_ml) / ed_volume_ml) * 100.0
    return ef


def calculate_all_cardiac_metrics(
    ed_mask_3d: np.ndarray,
    es_mask_3d: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
    myocardial_density: float = 1.05  # g/ml
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive cardiac metrics from ED and ES masks.

    Args:
        ed_mask_3d: End-diastolic 3D segmentation mask
        es_mask_3d: End-systolic 3D segmentation mask
        pixel_spacing: Physical spacing (x_mm, y_mm, z_mm)
        myocardial_density: Tissue density for mass calculation (g/ml, default=1.05)

    Returns:
        Dictionary with metrics for each cardiac structure:
        {
            'LV': {
                'ed_volume_ml': float,
                'es_volume_ml': float,
                'ejection_fraction_%': float,
                'stroke_volume_ml': float
            },
            'RV': {
                'ed_volume_ml': float,
                'es_volume_ml': float,
                'stroke_volume_ml': float
            },
            'MYO': {
                'ed_volume_ml': float,
                'es_volume_ml': float,
                'ed_mass_g': float,  # More clinically relevant for myocardium
                'es_mass_g': float,
                'average_mass_g': float
            }
        }

    Note:
        - Stroke volume is only calculated for ventricles (LV/RV cavities)
        - For myocardium (MYO), mass is reported instead of stroke volume,
          as the myocardial wall thickens during contraction (not a cavity)

    Example:
        >>> metrics = calculate_all_cardiac_metrics(ed_gt, es_gt, spacing)
        >>> lv_ef = metrics['LV']['ejection_fraction_%']
        >>> myo_mass = metrics['MYO']['average_mass_g']
        >>> print(f"LV EF: {lv_ef:.1f}%, Myocardial mass: {myo_mass:.1f}g")
    """
    structures = {
        'RV': 1,   # Right ventricle
        'MYO': 2,  # Myocardium
        'LV': 3    # Left ventricle
    }

    metrics = {}

    for name, class_id in structures.items():
        ed_vol = calculate_cardiac_volume(ed_mask_3d, pixel_spacing, class_id)
        es_vol = calculate_cardiac_volume(es_mask_3d, pixel_spacing, class_id)

        ed_volume_ml = ed_vol['volume_ml']
        es_volume_ml = es_vol['volume_ml']

        if name == 'MYO':
            # For myocardium, report mass instead of stroke volume
            # Myocardial wall thickens during systole, so volume change is not meaningful
            ed_mass_g = ed_volume_ml * myocardial_density
            es_mass_g = es_volume_ml * myocardial_density
            avg_mass_g = (ed_mass_g + es_mass_g) / 2.0

            metrics[name] = {
                'ed_volume_ml': ed_volume_ml,
                'es_volume_ml': es_volume_ml,
                'ed_mass_g': ed_mass_g,
                'es_mass_g': es_mass_g,
                'average_mass_g': avg_mass_g,
            }
        else:
            # For ventricles, calculate stroke volume and ejection fraction
            stroke_volume = ed_volume_ml - es_volume_ml

            metrics[name] = {
                'ed_volume_ml': ed_volume_ml,
                'es_volume_ml': es_volume_ml,
                'stroke_volume_ml': stroke_volume,
            }

            # Ejection fraction only for LV
            if name == 'LV':
                ef = calculate_ejection_fraction(ed_volume_ml, es_volume_ml)
                metrics[name]['ejection_fraction_%'] = ef

    return metrics


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between prediction and ground truth.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        IoU score (0-1), where 1 is perfect overlap

    Example:
        >>> pred = np.array([[1, 1, 0], [1, 0, 0]])
        >>> gt = np.array([[1, 0, 0], [1, 1, 0]])
        >>> iou = compute_iou(pred, gt)
        >>> print(f"IoU: {iou:.3f}")
    """
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute pixel-wise classification accuracy.

    Args:
        pred_mask: Predicted segmentation mask with class labels
        gt_mask: Ground truth segmentation mask with class labels

    Returns:
        Pixel accuracy (0-1), percentage of correctly classified pixels

    Example:
        >>> pred = np.array([[1, 2, 0], [1, 0, 2]])
        >>> gt = np.array([[1, 2, 0], [1, 1, 2]])
        >>> acc = compute_pixel_accuracy(pred, gt)
        >>> print(f"Accuracy: {acc:.1%}")
    """
    correct_pixels = (pred_mask == gt_mask).sum()
    total_pixels = pred_mask.size

    return float(correct_pixels / total_pixels)


def compute_volume_difference_percent(
    pred_volume_ml: float,
    gt_volume_ml: float
) -> float:
    """
    Compute percentage difference between predicted and ground truth volumes.

    Clinical interpretation:
        - <5%: Excellent agreement
        - 5-10%: Good agreement
        - 10-20%: Moderate agreement
        - >20%: Poor agreement

    Args:
        pred_volume_ml: Predicted volume in milliliters
        gt_volume_ml: Ground truth volume in milliliters

    Returns:
        Absolute percentage difference

    Example:
        >>> pred_vol = 145.2  # ml
        >>> gt_vol = 150.0    # ml
        >>> diff = compute_volume_difference_percent(pred_vol, gt_vol)
        >>> print(f"Volume error: {diff:.1f}%")
    """
    if gt_volume_ml == 0:
        return 100.0 if pred_volume_ml > 0 else 0.0

    return abs(pred_volume_ml - gt_volume_ml) / gt_volume_ml * 100.0


def compute_per_class_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_labels: Dict[int, str],
    pixel_spacing: Optional[Tuple[float, ...]] = None,
    skip_background: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive metrics for each class in multi-class segmentation.

    Based on patterns from acdc_visualization.ipynb for structure-specific analysis.

    Args:
        pred_mask: Predicted multi-class segmentation
        gt_mask: Ground truth multi-class segmentation
        class_labels: Mapping from label ID to structure name
                     e.g., {0: 'Background', 1: 'RV', 2: 'MYO', 3: 'LV'}
        pixel_spacing: Physical spacing for volume calculation (optional)
        skip_background: Whether to skip background class (label 0)

    Returns:
        Dict mapping structure name to metrics:
            - dice: Dice coefficient
            - iou: Intersection over Union
            - pixel_count_gt: Number of pixels in ground truth
            - pixel_count_pred: Number of pixels in prediction
            - volume_ml: Volume in milliliters (if pixel_spacing provided)
            - volume_diff_percent: Volume difference percentage

    Example:
        >>> labels = {0: 'Background', 1: 'RV', 2: 'MYO', 3: 'LV'}
        >>> metrics = compute_per_class_metrics(pred, gt, labels, (1.5, 1.5, 10.0))
        >>> print(f"LV Dice: {metrics['LV']['dice']:.3f}")
    """
    metrics = {}

    for label_id, structure_name in class_labels.items():
        # Skip background if requested
        if skip_background and label_id == 0:
            continue

        # Create binary masks for this class
        pred_binary = (pred_mask == label_id).astype(np.uint8)
        gt_binary = (gt_mask == label_id).astype(np.uint8)

        # Compute metrics
        dice = compute_dice_coefficient(pred_binary, gt_binary)
        iou = compute_iou(pred_binary, gt_binary)

        pixel_count_gt = gt_binary.sum()
        pixel_count_pred = pred_binary.sum()

        structure_metrics = {
            'dice': dice,
            'iou': iou,
            'pixel_count_gt': int(pixel_count_gt),
            'pixel_count_pred': int(pixel_count_pred),
        }

        # Add volume metrics if spacing provided
        if pixel_spacing is not None:
            # Calculate volumes
            voxel_volume_mm3 = np.prod(pixel_spacing)  # mm³ per voxel

            gt_volume_mm3 = pixel_count_gt * voxel_volume_mm3
            pred_volume_mm3 = pixel_count_pred * voxel_volume_mm3

            gt_volume_ml = gt_volume_mm3 / 1000.0  # Convert mm³ to ml
            pred_volume_ml = pred_volume_mm3 / 1000.0

            structure_metrics['volume_gt_ml'] = gt_volume_ml
            structure_metrics['volume_pred_ml'] = pred_volume_ml
            structure_metrics['volume_diff_percent'] = compute_volume_difference_percent(
                pred_volume_ml, gt_volume_ml
            )

        metrics[structure_name] = structure_metrics

    return metrics


def compute_all_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    class_labels: Dict[int, str],
    pixel_spacing: Optional[Tuple[float, ...]] = None,
    skip_background: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive segmentation quality metrics with summary statistics.

    This is a high-level function that combines per-class metrics with overall
    statistics, suitable for agent consumption and reporting.

    Based on workflow patterns from acdc_visualization.ipynb.

    Args:
        pred_mask: Predicted multi-class segmentation
        gt_mask: Ground truth multi-class segmentation
        class_labels: Mapping from label ID to structure name
        pixel_spacing: Physical spacing for volume calculation (optional)
        skip_background: Whether to skip background class

    Returns:
        Dictionary containing:
            - per_class: Metrics for each structure
            - overall: Aggregate metrics
                - mean_dice: Average Dice across all structures
                - mean_iou: Average IoU across all structures
                - pixel_accuracy: Overall pixel classification accuracy
                - best_structure: Structure with highest Dice
                - worst_structure: Structure with lowest Dice
            - metadata: Information about the analysis

    Example:
        >>> labels = {0: 'BG', 1: 'RV', 2: 'MYO', 3: 'LV'}
        >>> results = compute_all_metrics(pred, gt, labels, (1.5, 1.5, 10.0))
        >>> print(f"Mean Dice: {results['overall']['mean_dice']:.3f}")
        >>> print(f"Best: {results['overall']['best_structure']}")
    """
    # Compute per-class metrics
    per_class = compute_per_class_metrics(
        pred_mask, gt_mask, class_labels, pixel_spacing, skip_background
    )

    # Compute overall pixel accuracy (all classes)
    pixel_accuracy = compute_pixel_accuracy(pred_mask, gt_mask)

    # Aggregate statistics
    if per_class:
        dice_scores = [m['dice'] for m in per_class.values()]
        iou_scores = [m['iou'] for m in per_class.values()]

        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)

        # Find best and worst structures
        best_structure = max(per_class.items(), key=lambda x: x[1]['dice'])[0]
        worst_structure = min(per_class.items(), key=lambda x: x[1]['dice'])[0]
    else:
        mean_dice = 0.0
        mean_iou = 0.0
        best_structure = None
        worst_structure = None

    return {
        'per_class': per_class,
        'overall': {
            'mean_dice': float(mean_dice),
            'mean_iou': float(mean_iou),
            'pixel_accuracy': float(pixel_accuracy),
            'best_structure': best_structure,
            'worst_structure': worst_structure,
            'num_structures': len(per_class),
        },
        'metadata': {
            'pred_shape': pred_mask.shape,
            'gt_shape': gt_mask.shape,
            'has_volume_metrics': pixel_spacing is not None,
        }
    }
