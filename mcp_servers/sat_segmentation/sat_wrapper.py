"""
Wrapper around external SAT repository.
Uses SAT's official inference.py CLI interface for maximum reliability.

This approach calls SAT's inference.py as a subprocess rather than
trying to import and call SAT functions directly, which is more robust
to SAT's internal changes and dependencies.
"""

import sys
import os
import json
import tempfile
import subprocess
import logging
import socket
from pathlib import Path
from typing import List, Dict, Optional, Union

# Use anyio for async subprocess calls
import anyio

logger = logging.getLogger(__name__)

# Configuration from environment
SAT_REPO_PATH = Path(os.environ.get("SAT_REPO_PATH", "~/SAT")).expanduser()
CHECKPOINT_DIR = Path(os.environ.get("SAT_CHECKPOINT_DIR", f"{SAT_REPO_PATH}/checkpoints")).expanduser()

# Verify SAT repo exists
if not SAT_REPO_PATH.exists():
    raise RuntimeError(
        f"SAT repository not found at {SAT_REPO_PATH}. "
        f"Please run: bash scripts/setup_sat.sh"
    )


def get_free_port() -> int:
    """Get a truly free port for torchrun to avoid conflicts.

    Returns:
        int: Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _tail_text(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _summarize_sat_failure(stderr_str: str) -> str:
    if not stderr_str:
        return "SAT inference failed with unknown error."
    lower = stderr_str.lower()
    if "out of memory" in lower or "cuda out of memory" in lower:
        return (
            "SAT inference failed: CUDA out of memory. "
            "Free GPU memory or run on a less busy GPU. "
            f"Details:\n{_tail_text(stderr_str)}"
        )
    return _tail_text(stderr_str)


async def run_sat_inference_cli_async(
    image_path: str,
    structures: List[str],
    modality: str = "mri",
    output_dir: Optional[str] = None,
    model_variant: str = "nano",
    batchsize_3d: int = 1,
    dataset_name: str = "ACDC"
) -> Dict:
    """Async version: Run SAT inference using their official CLI interface.

    This uses anyio.run_process to avoid blocking the event loop during long-running
    SAT inference operations (which can take several minutes).

    Args:
        image_path: Path to NIfTI image file
        structures: List of structures to segment (e.g., ["left heart ventricle"])
        modality: Imaging modality ("mri", "ct", "pet")
        output_dir: Where to save results (temp dir if None)
        model_variant: "nano" or "pro"
        batchsize_3d: Batch size for 3D patches (1=24GB GPU, 2=36GB GPU for nano)
        dataset_name: Name for organizing outputs (default: "ACDC")

    Returns:
        Dictionary with output paths and metadata

    Raises:
        RuntimeError: If SAT inference fails
    """
    # Create temp jsonl file for SAT input format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        data = {
            "image": str(Path(image_path).absolute()),
            "label": structures,
            "modality": modality,
            "dataset": dataset_name
        }
        json.dump(data, f)
        f.write('\n')
        jsonl_path = f.name

    # Setup output directory
    if output_dir is None:
        # Use project outputs directory instead of /tmp
        project_root = Path(__file__).parent.parent.parent
        outputs_dir = project_root / "outputs" / "sat_mcp" / "segmentations"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix="sat_results_", dir=str(outputs_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model configuration
    if model_variant.lower() == "nano":
        vision_backbone = "UNET"
        checkpoint = CHECKPOINT_DIR / "Nano" / "nano.pth"
        text_checkpoint = CHECKPOINT_DIR / "Nano" / "nano_text_encoder.pth"
    elif model_variant.lower() == "pro":
        vision_backbone = "UNET-L"
        checkpoint = CHECKPOINT_DIR / "Pro" / "SAT_Pro.pth"
        text_checkpoint = CHECKPOINT_DIR / "Pro" / "text_encoder.pth"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}. Use 'nano' or 'pro'")

    # Verify checkpoints exist
    if not checkpoint.exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Please run: bash scripts/setup_sat.sh"
        )
    if not text_checkpoint.exists():
        raise RuntimeError(
            f"Text encoder not found: {text_checkpoint}\n"
            f"Please run: bash scripts/setup_sat.sh"
        )

    # Get a free port to avoid conflicts
    master_port = get_free_port()

    # Build SAT inference command
    # Use the same Python executable that's running this MCP server
    # This ensures we inherit the correct environment and dependencies
    current_python = sys.executable
    cmd = [
        current_python, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port", str(master_port),
        str(SAT_REPO_PATH / "inference.py"),
        "--rcd_dir", str(output_dir),
        "--datasets_jsonl", jsonl_path,
        "--vision_backbone", vision_backbone,
        "--checkpoint", str(checkpoint),
        "--text_encoder", "ours",
        "--text_encoder_checkpoint", str(text_checkpoint),
        "--max_queries", "256",
        "--batchsize_3d", str(batchsize_3d)
    ]

    logger.info(f"Starting SAT inference for {len(structures)} structures ({model_variant} model)")

    try:
        # Run SAT inference asynchronously using anyio
        # Ensure we use the correct Python environment by modifying PATH
        env = os.environ.copy()
        sat_env_bin = os.path.dirname(sys.executable)
        env['PATH'] = f"{sat_env_bin}:{env['PATH']}"
        
        with anyio.fail_after(300):  # 5 minute timeout
            result = await anyio.run_process(
                cmd,
                cwd=str(SAT_REPO_PATH),
                env=env,
                check=False, # Don't raise on non-zero exit
            )

        # Log subprocess output for debugging
        if result.stdout:
            stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
            logger.debug(f"SAT stdout:\n{stdout}")
        if result.stderr:
            stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
            logger.debug(f"SAT stderr:\n{stderr}")

        # Cleanup temp jsonl file
        os.unlink(jsonl_path)

        if result.returncode != 0:
            stderr_str = result.stderr.decode() if isinstance(result.stderr, bytes) else str(result.stderr)
            raise RuntimeError(
                f"SAT inference failed with code {result.returncode}: "
                f"{_summarize_sat_failure(stderr_str)}"
            )

        # Construct exact output file paths based on SAT's naming convention
        # SAT saves to: {output_dir}/{dataset_name}/seg_{sample_id}/{structure}.nii.gz
        # Handle .nii.gz files: need to remove both extensions
        img_path = Path(image_path)
        sample_id = img_path.name.replace('.nii.gz', '').replace('.nii', '')  # e.g., "patient001_frame01"

        seg_dir = output_dir / dataset_name / f"seg_{sample_id}"

        # Individual structure segmentation files
        seg_files = [str(seg_dir / f"{structure}.nii.gz") for structure in structures]

        # Combined segmentation file (all structures merged)
        combined_seg = str(output_dir / dataset_name / f"seg_{sample_id}.nii.gz")

        # Verify at least the combined segmentation was created
        if not Path(combined_seg).exists():
            raise RuntimeError(
                f"SAT inference completed but expected output not found: {combined_seg}\n"
                f"Check SAT logs for details."
            )

        return {
            "success": True,
            "structures": structures,
            "modality": modality,
            "model_variant": model_variant,
            "dataset_name": dataset_name,
            "input_image": str(image_path),
            "output_directory": str(output_dir),
            "segmentation_files": seg_files,
            "combined_segmentation": combined_seg,
            "num_files": len(seg_files),
            "message": f"Successfully segmented {len(structures)} structures. Output saved to {output_dir}",
            "primary_segmentation": combined_seg
        }

    except TimeoutError:
        os.unlink(jsonl_path)
        raise RuntimeError("SAT inference timed out after 5 minutes")
    except Exception as e:
        # Cleanup on error
        try:
            os.unlink(jsonl_path)
        except:
            pass
        logger.error(f"✗ SAT inference failed: {e}", exc_info=True)
        raise RuntimeError(f"SAT inference failed: {str(e)}")


async def run_sat_inference_cli_batch_async(
    image_paths: List[str],
    structures: Union[List[str], List[List[str]]],
    modality: str = "mri",
    output_dir: Optional[str] = None,
    model_variant: str = "nano",
    batchsize_3d: int = 1
) -> Dict:
    """Async version: Run SAT inference on multiple images in a single model load.

    This creates a multi-line JSONL file where each line represents one patient/image.
    SAT will load the model once and process all images sequentially, amortizing the
    expensive model loading time (~60s) across all images.

    Args:
        image_paths: List of paths to NIfTI image files
        structures: Either a single list of structures to use for all images,
                   or a list of structure lists (one per image)
        modality: Imaging modality ("mri", "ct", "pet")
        output_dir: Where to save results (temp dir if None)
        model_variant: "nano" or "pro"
        batchsize_3d: Batch size for 3D patches (1=24GB GPU, 2=36GB GPU for nano)

    Returns:
        Dictionary with batch results mapping image_path -> output info

    Example:
        # Same structures for all images
        result = await run_sat_inference_cli_batch_async(
            image_paths=["/path/img1.nii.gz", "/path/img2.nii.gz"],
            structures=["left ventricle", "myocardium"],
            modality="mri"
        )

        # For different modalities per image, call separately
    """
    num_images = len(image_paths)
    if num_images == 0:
        raise ValueError("image_paths cannot be empty")

    # Normalize structures: if single list, use for all images
    if isinstance(structures[0], str):
        structures_per_image = [structures] * num_images
    else:
        structures_per_image = structures
        if len(structures_per_image) != num_images:
            raise ValueError(f"structures list length ({len(structures_per_image)}) must match image_paths ({num_images})")

    # Normalize modalities: if single string, use for all images
    modalities_per_image = [modality] * num_images

    # Create multi-line JSONL file for batch processing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for img_path, structs, mod in zip(image_paths, structures_per_image, modalities_per_image):
            data = {
                "image": str(Path(img_path).absolute()),
                "label": structs,
                "modality": mod,
                "dataset": "ACDC"  # Can be any name
            }
            json.dump(data, f)
            f.write('\n')
        jsonl_path = f.name

    # Setup output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        outputs_dir = project_root / "outputs" / "sat_mcp" / "segmentations"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix="sat_batch_", dir=str(outputs_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model configuration
    if model_variant.lower() == "nano":
        vision_backbone = "UNET"
        checkpoint = CHECKPOINT_DIR / "Nano" / "nano.pth"
        text_checkpoint = CHECKPOINT_DIR / "Nano" / "nano_text_encoder.pth"
    elif model_variant.lower() == "pro":
        vision_backbone = "UNET-L"
        checkpoint = CHECKPOINT_DIR / "Pro" / "SAT_Pro.pth"
        text_checkpoint = CHECKPOINT_DIR / "Pro" / "text_encoder.pth"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}. Use 'nano' or 'pro'")

    # Verify checkpoints exist
    if not checkpoint.exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Please run: bash scripts/setup_sat.sh"
        )
    if not text_checkpoint.exists():
        raise RuntimeError(
            f"Text encoder not found: {text_checkpoint}\n"
            f"Please run: bash scripts/setup_sat.sh"
        )

    # Get a free port
    master_port = get_free_port()

    # Build SAT inference command
    current_python = sys.executable
    cmd = [
        current_python, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port", str(master_port),
        str(SAT_REPO_PATH / "inference.py"),
        "--rcd_dir", str(output_dir),
        "--datasets_jsonl", jsonl_path,
        "--vision_backbone", vision_backbone,
        "--checkpoint", str(checkpoint),
        "--text_encoder", "ours",
        "--text_encoder_checkpoint", str(text_checkpoint),
        "--max_queries", "256",
        "--batchsize_3d", str(batchsize_3d)
    ]

    # Dynamic timeout: 5 min base + 2 min per image
    timeout_seconds = 300 + (120 * num_images)

    logger.info(f"Starting SAT batch inference for {num_images} images ({model_variant} model, timeout={timeout_seconds}s)")

    try:
        # Run SAT inference asynchronously
        env = os.environ.copy()
        sat_env_bin = os.path.dirname(sys.executable)
        env['PATH'] = f"{sat_env_bin}:{env['PATH']}"

        with anyio.fail_after(timeout_seconds):
            result = await anyio.run_process(
                cmd,
                cwd=str(SAT_REPO_PATH),
                env=env,
                check=False,
            )

        # Log subprocess output
        if result.stdout:
            stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
            logger.debug(f"SAT stdout:\n{stdout}")
        if result.stderr:
            stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
            logger.debug(f"SAT stderr:\n{stderr}")

        # Cleanup temp jsonl file
        os.unlink(jsonl_path)

        if result.returncode != 0:
            stderr_str = result.stderr.decode() if isinstance(result.stderr, bytes) else str(result.stderr)
            raise RuntimeError(
                f"SAT inference failed with code {result.returncode}: "
                f"{_summarize_sat_failure(stderr_str)}"
            )

        # Parse outputs for each image
        # SAT creates output structure: output_dir/ACDC/seg_{sample_id}/structure.nii.gz
        batch_results = {}
        for img_path in image_paths:
            img_path_obj = Path(img_path)
            sample_id = img_path_obj.name.replace('.nii.gz', '').replace('.nii', '')  # e.g., "patient001_frame01"
                
            seg_dir = output_dir / "ACDC" / f"seg_{sample_id}"

            if seg_dir.exists():
                seg_files = list(seg_dir.glob("*.nii.gz"))
                batch_results[str(img_path)] = {
                    "success": True,
                    "output_directory": str(seg_dir),
                    "num_files": len(seg_files),
                    "structures_found": [f.stem.replace(f"{img_path_obj.stem}_", "") for f in seg_files],
                    "segmentation_files": [str(f) for f in sorted(seg_files)]  # All segmentation files for auto-save
                }
            else:
                batch_results[str(img_path)] = {
                    "success": False,
                    "error": f"Output directory not found: {seg_dir}",
                    "output_directory": str(seg_dir),
                    "num_files": 0,
                    "structures_found": [],
                    "segmentation_files": []
                }

        # Create summary statistics
        total_files = sum(result["num_files"] for result in batch_results.values())
        successful_images = sum(1 for result in batch_results.values() if result["success"])
        all_structures = set()
        for result in batch_results.values():
            all_structures.update(result["structures_found"])

        return {
            "success": True,
            "batch_size": num_images,
            "model_variant": model_variant,
            "output_directory": str(output_dir),
            "summary": {
                "successful_images": successful_images,
                "total_files_generated": total_files,
                "unique_structures": sorted(list(all_structures))
            },
            "results": batch_results,
            "message": f"Successfully processed {successful_images}/{num_images} images in batch. Generated {total_files} segmentation files. Output saved to {output_dir}"
        }

    except TimeoutError:
        os.unlink(jsonl_path)
        raise RuntimeError(f"SAT batch inference timed out after {timeout_seconds} seconds ({num_images} images)")
    except Exception as e:
        try:
            os.unlink(jsonl_path)
        except:
            pass
        logger.error(f"✗ SAT batch inference failed: {e}", exc_info=True)
        raise RuntimeError(f"SAT batch inference failed: {str(e)}")


def run_sat_inference_cli(
    image_path: str,
    structures: List[str],
    modality: str = "mri",
    output_dir: Optional[str] = None,
    model_variant: str = "nano",
    batchsize_3d: int = 1,
    dataset_name: str = "ACDC"
) -> Dict:
    """Run SAT inference using their official CLI interface.

    This uses SAT's inference.py with torchrun for reliability.

    Args:
        image_path: Path to NIfTI image file
        structures: List of structures to segment (e.g., ["left heart ventricle"])
        modality: Imaging modality ("mri", "ct", "pet")
        output_dir: Where to save results (temp dir if None)
        model_variant: "nano" or "pro"
        batchsize_3d: Batch size for 3D patches (1=24GB GPU, 2=36GB GPU for nano)
        dataset_name: Name for organizing outputs (default: "ACDC")

    Returns:
        Dictionary with output paths and metadata

    Raises:
        RuntimeError: If SAT inference fails
    """
    # Create temp jsonl file for SAT input format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        data = {
            "image": str(Path(image_path).absolute()),
            "label": structures,
            "modality": modality,
            "dataset": dataset_name
        }
        json.dump(data, f)
        f.write('\n')
        jsonl_path = f.name

    # Setup output directory
    if output_dir is None:
        # Use project outputs directory instead of /tmp
        project_root = Path(__file__).parent.parent.parent
        outputs_dir = project_root / "outputs" / "sat_mcp" / "segmentations"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix="sat_results_", dir=str(outputs_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model configuration
    if model_variant.lower() == "nano":
        vision_backbone = "UNET"
        checkpoint = CHECKPOINT_DIR / "Nano" / "nano.pth"
        text_checkpoint = CHECKPOINT_DIR / "Nano" / "nano_text_encoder.pth"
    elif model_variant.lower() == "pro":
        vision_backbone = "UNET-L"
        checkpoint = CHECKPOINT_DIR / "Pro" / "SAT_Pro.pth"
        text_checkpoint = CHECKPOINT_DIR / "Pro" / "text_encoder.pth"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}. Use 'nano' or 'pro'")

    # Verify checkpoints exist
    if not checkpoint.exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Please run: bash scripts/setup_sat.sh"
        )
    if not text_checkpoint.exists():
        raise RuntimeError(
            f"Text encoder not found: {text_checkpoint}\n"
            f"Please run: bash scripts/setup_sat.sh"
        )

    # Get a free port to avoid conflicts
    master_port = get_free_port()

    # Build SAT inference command
    # Use torchrun as SAT expects (even with single GPU)
    # Use the same Python executable that's running this MCP server
    # This ensures we inherit the correct environment and dependencies
    current_python = sys.executable
    cmd = [
        current_python, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port", str(master_port),
        str(SAT_REPO_PATH / "inference.py"),
        "--rcd_dir", str(output_dir),
        "--datasets_jsonl", jsonl_path,
        "--vision_backbone", vision_backbone,
        "--checkpoint", str(checkpoint),
        "--text_encoder", "ours",
        "--text_encoder_checkpoint", str(text_checkpoint),
        "--max_queries", "256",
        "--batchsize_3d", str(batchsize_3d)
    ]

    logger.info(f"Starting SAT inference for {len(structures)} structures ({model_variant} model)")

    try:
        # Run SAT inference
        # Ensure we use the correct Python environment by modifying PATH
        env = os.environ.copy()
        sat_env_bin = os.path.dirname(sys.executable)
        env['PATH'] = f"{sat_env_bin}:{env['PATH']}"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SAT_REPO_PATH),  # Run from SAT directory
            env=env,
            timeout=300  # 5 minute timeout
        )

        # Log subprocess output for debugging
        if result.stdout:
            logger.debug(f"SAT stdout:\n{result.stdout}")
        if result.stderr:
            logger.debug(f"SAT stderr:\n{result.stderr}")

        # Cleanup temp jsonl file
        os.unlink(jsonl_path)

        if result.returncode != 0:
            stderr_str = result.stderr.decode() if isinstance(result.stderr, bytes) else str(result.stderr)
            raise RuntimeError(
                f"SAT inference failed with code {result.returncode}: "
                f"{_summarize_sat_failure(stderr_str)}"
            )

        # Construct exact output file paths based on SAT's naming convention
        # Handle .nii.gz files: need to remove both extensions
        img_path = Path(image_path)
        sample_id = img_path.name.replace('.nii.gz', '').replace('.nii', '')
        seg_dir = output_dir / dataset_name / f"seg_{sample_id}"

        # Individual structure segmentation files
        seg_files = [str(seg_dir / f"{structure}.nii.gz") for structure in structures]

        # Combined segmentation file
        combined_seg = str(output_dir / dataset_name / f"seg_{sample_id}.nii.gz")

        # Verify at least the combined segmentation was created
        if not Path(combined_seg).exists():
            raise RuntimeError(
                f"SAT inference completed but expected output not found: {combined_seg}\n"
                f"Check SAT logs for details."
            )

        return {
            "success": True,
            "output_directory": str(output_dir),
            "segmentation_files": seg_files,
            "combined_segmentation": combined_seg,
            "num_structures": len(structures),
            "structures": structures,
            "dataset_name": dataset_name,
            "model_variant": model_variant
        }

    except subprocess.TimeoutExpired:
        os.unlink(jsonl_path)
        raise RuntimeError("SAT inference timed out after 5 minutes")
    except Exception as e:
        # Cleanup on error
        try:
            os.unlink(jsonl_path)
        except:
            pass
        raise RuntimeError(f"SAT inference failed: {str(e)}")


def check_sat_installation() -> Dict:
    """Check if SAT is properly installed and configured.

    Returns:
        Dictionary with installation status
    """
    status = {
        "sat_repo_path": str(SAT_REPO_PATH),
        "sat_repo_exists": SAT_REPO_PATH.exists(),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_dir_exists": CHECKPOINT_DIR.exists(),
        "checkpoints": {}
    }

    # Check for checkpoints
    if CHECKPOINT_DIR.exists():
        # Nano checkpoints
        nano_model = CHECKPOINT_DIR / "Nano" / "nano.pth"
        nano_encoder = CHECKPOINT_DIR / "Nano" / "nano_text_encoder.pth"
        status["checkpoints"]["nano"] = {
            "model": nano_model.exists(),
            "text_encoder": nano_encoder.exists(),
            "complete": nano_model.exists() and nano_encoder.exists()
        }

        # Pro checkpoints
        pro_model = CHECKPOINT_DIR / "Pro" / "SAT_Pro.pth"
        pro_encoder = CHECKPOINT_DIR / "Pro" / "text_encoder.pth"
        status["checkpoints"]["pro"] = {
            "model": pro_model.exists(),
            "text_encoder": pro_encoder.exists(),
            "complete": pro_model.exists() and pro_encoder.exists()
        }

    # Check if inference.py exists
    inference_py = SAT_REPO_PATH / "inference.py"
    status["inference_script_exists"] = inference_py.exists()

    # Overall status
    status["ready"] = (
        status["sat_repo_exists"] and
        status["inference_script_exists"] and
        status["checkpoint_dir_exists"] and
        any(cp["complete"] for cp in status["checkpoints"].values())
    )

    return status
