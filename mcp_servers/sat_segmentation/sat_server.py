#!/usr/bin/env python3
"""
SAT MCP Server - Exposes SAT segmentation via Model Context Protocol.

This server wraps the external SAT repository and exposes it as MCP tools
that VERITAS agents can use for medical image segmentation.

Usage:
    python sat_server.py

Environment variables:
    SAT_REPO_PATH: Path to SAT repository (default: ~/SAT)
    SAT_CHECKPOINT_DIR: Path to checkpoints (default: ~/SAT/checkpoints)
    CUDA_VISIBLE_DEVICES: GPU selection (default: 0)
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict

# MCP imports
from mcp.server import Server
from mcp.types import Tool, TextContent

# Use anyio for thread pool operations
import anyio

# SAT wrapper (uses CLI interface to external repo)
from sat_wrapper import (
    run_sat_inference_cli,
    run_sat_inference_cli_async,
    run_sat_inference_cli_batch_async,
    check_sat_installation
)

# Structure catalog for agent discovery
from structure_catalog import (
    get_structures_by_category,
    search_structures,
    get_all_categories,
    format_catalog_for_agents,
    STRUCTURE_CATALOG
)

# Database integration for automatic result saving
try:
    # Import ResultsDatabase directly without loading the veritas package
    # (which has dependencies like openai that aren't in the SAT environment)
    import importlib.util
    results_db_path = Path(__file__).parent.parent.parent / "src" / "veritas" / "vision" / "results_db.py"
    spec = importlib.util.spec_from_file_location("results_db", results_db_path)
    results_db_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(results_db_module)
    ResultsDatabase = results_db_module.ResultsDatabase

    DATABASE_AVAILABLE = True
    _database_import_error = None
    # Initialize database with optional path parameter
    _results_db_cache = {}
    def _get_db(results_database):
        """Get or create ResultsDatabase instance.

        Args:
            results_database: Database path for storing results.

        Returns:
            ResultsDatabase instance
        """
        if results_database not in _results_db_cache:
            _results_db_cache[results_database] = ResultsDatabase(results_database)
        return _results_db_cache[results_database]
except Exception as e:
    DATABASE_AVAILABLE = False
    _database_import_error = str(e)
    def _get_db():
        return None

# Configure logging to file only (not stderr) to avoid corrupting MCP JSON-RPC protocol
log_file = Path(__file__).parent.parent.parent / "tmp" / "sat_mcp_server.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        # Don't add StreamHandler - it would corrupt JSON-RPC on stderr!
    ]
)
logger = logging.getLogger(__name__)


def _normalize_structure_name(name: str) -> str:
    alias_map = {
        "left ventricle": "left heart ventricle",
        "lv": "left heart ventricle",
        "right ventricle": "right heart ventricle",
        "rv": "right heart ventricle",
        "myo": "myocardium",
    }
    normalized = name.strip().lower()
    return alias_map.get(normalized, name)


def _normalize_structures(structures):
    if not structures:
        return structures
    if isinstance(structures, list) and all(isinstance(s, str) for s in structures):
        return [_normalize_structure_name(s) for s in structures]
    if isinstance(structures, list) and all(isinstance(s, list) for s in structures):
        return [[_normalize_structure_name(s) for s in sub] for sub in structures]
    return structures
logger.info(f"Logging to {log_file}")

# Log database availability status
if DATABASE_AVAILABLE:
    logger.info("Database integration enabled - auto-save active")
else:
    logger.info(f"Database integration disabled - auto-save inactive (reason: {_database_import_error})")

# Create MCP server
server = Server("sat-segmentation")

# Concurrency limiter to prevent multiple GPU-intensive inferences from running simultaneously
# This prevents GPU OOM errors. Set to 1 for safety on 24GB GPUs.
# Future: Could increase this limit or handle batching for dataset-level inference
_inference_limiter = anyio.CapacityLimiter(1)

# Dataset loader cache for resolving identifiers
_dataset_loaders = {}

def _get_dataset_root() -> str | None:
    """Resolve dataset root from environment."""
    return (
        os.environ.get("DATASET_PATH")
        or os.environ.get("ACDC_DATA_ROOT")
        or os.environ.get("ACDC_ROOT")
    )

def _get_acdc_loader():
    """Get or create ACDC dataset loader instance."""
    if 'acdc' not in _dataset_loaders:
        # Import ACDCDatasetLoader directly from files to avoid triggering veritas package initialization
        # (which would import langchain and other heavy dependencies not available in MCP server env)
        try:
            import sys
            import importlib.util

            # Add veritas src to path
            veritas_src = Path(__file__).parent.parent.parent / "src"
            if str(veritas_src) not in sys.path:
                sys.path.insert(0, str(veritas_src))

            # First, load base module (contains MedicalDatasetLoader, MedicalImage)
            base_path = veritas_src / "veritas" / "vision" / "base.py"
            base_spec = importlib.util.spec_from_file_location("veritas.vision.base", base_path)
            base_module = importlib.util.module_from_spec(base_spec)
            sys.modules["veritas.vision.base"] = base_module  # Register in sys.modules
            base_spec.loader.exec_module(base_module)

            # Now load acdc_loader module (it will find base via sys.modules)
            acdc_loader_path = veritas_src / "veritas" / "vision" / "datasets" / "acdc_loader.py"
            acdc_spec = importlib.util.spec_from_file_location(
                "veritas.vision.datasets.acdc_loader",
                acdc_loader_path
            )
            acdc_module = importlib.util.module_from_spec(acdc_spec)
            sys.modules["veritas.vision.datasets.acdc_loader"] = acdc_module  # Register in sys.modules
            acdc_spec.loader.exec_module(acdc_module)

            _dataset_loaders['acdc'] = acdc_module.ACDCDatasetLoader(
                data_root=_get_dataset_root(),
                split="all"
            )
        except Exception as e:
            logger.error(f"Failed to load ACDC dataset loader: {e}", exc_info=True)
            raise
    return _dataset_loaders['acdc']


def _get_manifest_loader(dataset_name: str):
    """Get or create manifest dataset loader instance."""
    key = f"manifest:{dataset_name.lower()}"
    if key not in _dataset_loaders:
        data_root = _get_dataset_root()
        if not data_root:
            raise ValueError(
                "Dataset root not configured. Set DATASET_PATH, "
                "ACDC_DATA_ROOT, or ACDC_ROOT."
            )
        try:
            import sys
            import importlib.util

            veritas_src = Path(__file__).parent.parent.parent / "src"
            if str(veritas_src) not in sys.path:
                sys.path.insert(0, str(veritas_src))

            manifest_loader_path = (
                veritas_src
                / "veritas"
                / "vision"
                / "datasets"
                / "manifest_loader.py"
            )
            manifest_spec = importlib.util.spec_from_file_location(
                "veritas.vision.datasets.manifest_loader",
                manifest_loader_path
            )
            manifest_module = importlib.util.module_from_spec(manifest_spec)
            sys.modules[
                "veritas.vision.datasets.manifest_loader"
            ] = manifest_module
            manifest_spec.loader.exec_module(manifest_module)

            _dataset_loaders[key] = manifest_module.ManifestDatasetLoader(
                data_root=data_root,
                dataset_name=dataset_name
            )
        except Exception as e:
            logger.error(
                f"Failed to load manifest dataset loader: {e}",
                exc_info=True
            )
            raise
    return _dataset_loaders[key]


def resolve_dataset_identifier(identifier: str) -> str:
    """Resolve dataset-relative identifier to actual file path.

    Supports format: dataset:patient_id:phase
    Example: acdc:patient001:ED -> /path/to/ACDC/database/training/patient001/patient001_frame01.nii.gz

    Args:
        identifier: Dataset-relative identifier or regular file path

    Returns:
        Resolved file path
    """
    # If it's already a valid path, return as-is
    if Path(identifier).exists():
        return identifier

    # Check if it's a dataset identifier (format: dataset:patient:phase)
    if ':' in identifier:
        parts = identifier.split(':')
        if len(parts) == 3:
            dataset, patient_id, phase = parts

            if dataset.lower() == 'acdc':
                loader = _get_acdc_loader()
                # Get patient metadata to find the correct frame
                metadata = loader.get_patient_metadata(patient_id)

                # Determine which frame to use based on phase
                if phase.upper() == 'ED':
                    frame_num = metadata['ed_frame']
                elif phase.upper() == 'ES':
                    frame_num = metadata['es_frame']
                else:
                    raise ValueError(f"Unknown phase '{phase}' for ACDC dataset. Use 'ED' or 'ES'.")

                # Build the full path
                split = loader.get_patient_split(patient_id)
                patient_dir = loader.data_root / split / patient_id
                image_path = patient_dir / f"{patient_id}_frame{frame_num:02d}.nii.gz"

                logger.info(f"Resolved identifier '{identifier}' -> {image_path}")
                return str(image_path)

            loader = _get_manifest_loader(dataset)
            resolved = loader.resolve_identifier(identifier)
            logger.info(f"Resolved identifier '{identifier}' -> {resolved}")
            return resolved

    # If we get here, it's neither a valid path nor a recognized identifier
    return identifier  # Return as-is and let validation fail later


def extract_patient_info_from_path(image_path):
    """Extract patient ID and cardiac phase from image path."""
    path = Path(image_path)
    
    # Try to extract patient ID from filename or parent directory
    # Common patterns: patient001_ED.nii.gz, patient001/ED/image.nii.gz, etc.
    patient_id = None
    cardiac_phase = None
    
    # Check filename for patterns
    filename = path.stem.replace('.nii', '')  # Remove .nii.gz -> .nii -> base
    
    # Pattern 1: patient001_ED, patient001_ES
    if '_' in filename:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower().startswith('patient'):
                patient_id = part
            if part.upper() in ['ED', 'ES']:
                cardiac_phase = part.upper()
    
    # Pattern 2: Check parent directories
    if not patient_id or not cardiac_phase:
        path_parts = path.parts
        for part in path_parts:
            if not patient_id and part.lower().startswith('patient'):
                patient_id = part
            if not cardiac_phase and part.upper() in ['ED', 'ES']:
                cardiac_phase = part.upper()
    
    # Fallback: use filename as patient_id if no pattern found
    if not patient_id:
        patient_id = filename.split('_')[0] if '_' in filename else filename
    
    return patient_id, cardiac_phase


async def save_segmentation_to_database(image_path, mask_path, structures, modality, model_variant, processing_time=None, results_database=None):
    """Save segmentation result to database if database is available.

    Args:
        image_path: Path to source medical image
        mask_path: Path to segmentation mask file
        structures: List of segmented structures
        modality: Imaging modality (mri, ct, pet)
        model_variant: SAT model variant (nano, pro)
        processing_time: Optional processing time in seconds
        results_database: Optional database path. If None, uses default.

    Returns:
        Result ID if saved successfully, None otherwise
    """
    if not DATABASE_AVAILABLE:
        logger.info("Database not available - skipping auto-save")
        return None

    try:
        # Extract patient information
        patient_id, cardiac_phase = extract_patient_info_from_path(image_path)

        # Prepare metadata
        metadata = {
            "structures": structures,
            "modality": modality,
            "model_variant": model_variant,
            "source_image": str(image_path),
            "cardiac_phase": cardiac_phase or "unknown",
            "processing_time": processing_time,
            "sat_version": "mcp_integration",
            "auto_saved": True
        }

        # Determine model name for database
        model_name = f"sat_{model_variant}"
        if cardiac_phase:
            model_name += f"_{cardiac_phase.lower()}"

        # Save to database
        db = _get_db(results_database)
        result_id = await anyio.to_thread.run_sync(
            lambda: db.save_segmentation_result(
                patient_id=patient_id,
                model_name=model_name,
                mask_path=mask_path,
                metadata=metadata
            )
        )
        
        logger.info(f"Auto-saved segmentation to database: {result_id}")
        return result_id
        
    except Exception as e:
        logger.error(f"Failed to auto-save segmentation to database: {e}")
        return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for medical image segmentation.

    Returns:
        List of available tools
    """
    return [
        Tool(
            name="segment_medical_structure",
            description=(
                "Segment anatomical structures in medical images using SAT foundation model. "
                "Uses official SAT-DS terminology. "
                "Supports 100+ structures across multiple organ systems: "
                "- Cardiac: myocardium, left/right heart ventricle, left/right atrium, aorta, pulmonary artery "
                "- Abdominal: liver, spleen, kidney, pancreas, stomach, gallbladder, intestine "
                "- Vascular: aorta, inferior vena cava, portal vein, hepatic vessel "
                "- Spine: cervical/thoracic/lumbar vertebrae, intervertebral discs "
                "- Brain: brain tumor, necrotic core, brain edema "
                "- Other: prostate, urinary bladder, bone, trachea, and more. "
                "Returns paths to segmentation files. For multiple images, use segment_structures_batch for better performance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to medical image (NIfTI format: .nii or .nii.gz)"
                    },
                    "structures": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of structures to segment. Can be a single structure or multiple structures. "
                            "Examples: ['left heart ventricle'], ['myocardium', 'left heart atrium'], etc. "
                            "Use official SAT-DS terminology for best results."
                        )
                    },
                    "modality": {
                        "type": "string",
                        "description": "Imaging modality",
                        "enum": ["mri", "ct", "pet"],
                        "default": "mri"
                    },
                    "model_variant": {
                        "type": "string",
                        "description": "SAT model variant",
                        "enum": ["nano", "pro"],
                        "default": "nano"
                    },
                    "results_database": {
                        "type": "string",
                        "description": "Database path for storing segmentation results"
                    }
                },
                "required": ["image_path", "structures", "results_database"]
            }
        ),
        Tool(
            name="segment_structures_batch",
            description=(
                "Segment anatomical structures in MULTIPLE medical images using SAT foundation model in a single batch. "
                "This is significantly faster than processing images one-by-one because it loads the model once "
                "and processes all images sequentially. Ideal for cohort studies or processing multiple patients. "
                "Supports 100+ anatomical structures across multiple organ systems: "
                "- Cardiac: myocardium, left/right heart ventricle, left/right atrium, aorta, pulmonary artery "
                "- Abdominal: liver, spleen, kidney, pancreas, stomach, gallbladder, intestine "
                "- Vascular: aorta, inferior vena cava, portal vein, hepatic vessel "
                "- Spine: cervical/thoracic/lumbar vertebrae, intervertebral discs "
                "- Brain: brain tumor, necrotic core, brain edema "
                "- Other: prostate, urinary bladder, bone, trachea, and more. "
                "You can segment different structures for different images in the same batch."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of paths to medical images (NIfTI format: .nii or .nii.gz)"
                    },
                    "structures": {
                        "description": (
                            "Structures to segment. Two formats supported:\n"
                            "Format 1 - Same structures for ALL images (recommended for uniform datasets):\n"
                            "  ['liver', 'spleen', 'kidney']\n"
                            "Format 2 - Different structures per image (if images need different targets):\n"
                            "  [['liver'], ['spleen', 'kidney'], ['liver', 'spleen']]\n"
                            "  (Must have one sublist per image)\n\n"
                            "For cardiac datasets, typically use Format 1 with: ['left heart ventricle', 'right heart ventricle', 'myocardium']"
                        ),
                        "oneOf": [
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Single list of structures to use for ALL images"
                            },
                            {
                                "type": "array", 
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "description": "List of structure lists, one per image"
                            }
                        ]
                    },
                    "modality": {
                        "description": "Imaging modality for all images in the batch",
                        "type": "string",
                        "enum": ["mri", "ct", "pet"],
                        "default": "mri"
                    },
                    "model_variant": {
                        "type": "string",
                        "description": "SAT model variant",
                        "enum": ["nano", "pro"],
                        "default": "nano"
                    },
                    "results_database": {
                        "type": "string",
                        "description": "Database path for storing segmentation results"
                    }
                },
                "required": ["image_paths", "structures", "results_database"]
            }
        ),
        Tool(
            name="list_available_structures",
            description=(
                "Discover available anatomical structures that SAT can segment. "
                "Useful for exploring the 100+ supported structures across different organ systems. "
                "Can filter by category (cardiac, abdominal, vascular, spine, brain, etc.) or search by keyword. "
                "Use category='all' to return the full catalog."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Filter by organ system category. Options: "
                            "cardiac, abdominal, vascular, spine, brain, urogenital, endocrine, "
                            "musculoskeletal, respiratory, liver_segments, all"
                        ),
                        "enum": [
                            "cardiac", "abdominal", "vascular", "spine", "brain",
                            "urogenital", "endocrine", "musculoskeletal", "respiratory", "liver_segments", "all"
                        ]
                    },
                    "search": {
                        "type": "string",
                        "description": "Search for structures containing this keyword (e.g., 'kidney', 'ventricle', 'tumor')"
                    }
                }
            }
        ),
        Tool(
            name="check_sat_status",
            description=(
                "Check if SAT model is installed and ready for inference. "
                "Verifies SAT repository, checkpoints, and configuration."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]
    logger.info("list_tools() returning 4 tools successfully")


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with args: {arguments}")

    try:
        if name == "segment_medical_structure":
            return await segment_medical_structure(**arguments)
        elif name == "segment_structures_batch":
            return await segment_structures_batch(**arguments)
        elif name == "list_available_structures":
            return await list_available_structures(**arguments)
        elif name == "check_sat_status":
            return await check_sat_status()
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"success": False, "error": str(e)}, indent=2)
        )]


async def segment_medical_structure(
    image_path: str,
    structures: list,
    results_database: str,
    modality: str = "mri",
    model_variant: str = "nano"
) -> list[TextContent]:
    """Execute SAT segmentation on anatomical structures using CLI interface.

    Args:
        image_path: Path to NIfTI image
        structures: List of structure names to segment (can be single or multiple)
        results_database: Database path for storing segmentation results
        modality: Imaging modality (mri, ct, pet)
        model_variant: Model variant (nano or pro)
    """

    # Validate structures parameter
    if not structures or not isinstance(structures, list):
        raise ValueError("'structures' must be a non-empty list of structure names")

    structures_list = _normalize_structures(structures)

    logger.info(f"segment_medical_structure called: {len(structures_list)} structures, modality={modality}, model={model_variant}")

    try:
        # Resolve dataset identifier to actual path if needed
        image_path = resolve_dataset_identifier(image_path)
        logger.info(f"→ Resolved image path: {image_path}")

        # Normalize modality to lowercase and validate
        modality = modality.lower() if modality else "mri"
        valid_modalities = ["mri", "ct", "pet"]
        if modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{modality}'. Must be one of: {valid_modalities}")
        logger.info(f"→ Normalized modality: {modality}")

        # Validate image path
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            logger.error(f"✗ Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Check for existing results before running inference
        if DATABASE_AVAILABLE:
            patient_id, cardiac_phase = extract_patient_info_from_path(image_path)
            model_name = f"sat_{model_variant}"
            if cardiac_phase:
                model_name += f"_{cardiac_phase.lower()}"
            
            logger.info(f"→ Checking for existing results: {patient_id}, {model_name}")

            try:
                db = _get_db(results_database)
                existing = await anyio.to_thread.run_sync(
                    lambda: db.find_existing_result(
                        patient_id,
                        model_name,
                        structures_list
                    )
                )
                
                if existing:
                    logger.info(f"✓ Found existing result, returning cached segmentation")
                    
                    # Build response from existing result
                    response = {
                        "success": True,
                        "structures": structures_list,
                        "modality": modality,
                        "model_variant": model_variant,
                        "input_image": str(image_path),
                        "output_directory": existing.get("result_dir", ""),
                        "segmentation_files": [existing.get("mask_path", "")],
                        "num_files": 1,
                        "processing_time": 0.0,  # No processing needed
                        "database_result_id": existing.get("result_id", ""),
                        "auto_saved": True,
                        "cached_result": True,
                        "message": f"Returned existing segmentation for {len(structures_list)} structures (cached result)"
                    }
                    
                    return [TextContent(type="text", text=json.dumps(response, indent=2))]
                    
            except Exception as e:
                logger.warning(f"Failed to check for existing results: {e}, proceeding with inference")

        # No existing result found, proceed with SAT inference

        # Determine output directory from results_database
        output_dir_for_sat = str(Path(results_database) / "outputs" / "sat_results")

        # Run SAT inference via CLI (async with concurrency limiting)
        # Use concurrency limiter to prevent multiple simultaneous GPU inferences
        async with _inference_limiter:
            import time
            start_time = time.time()

            result = await run_sat_inference_cli_async(
                image_path=str(image_path),
                structures=structures_list,  # SAT accepts list of structures
                modality=modality,
                model_variant=model_variant,
                output_dir=output_dir_for_sat,  # Derive from database path
                batchsize_3d=1  # Safe default for 24GB GPU
            )

            processing_time = time.time() - start_time

        # Parse output
        output_dir = Path(result["output_directory"])
        seg_files = result["segmentation_files"]

        logger.info(f"Segmentation complete: {len(seg_files)} files in {output_dir}")

        # Auto-save to database if available
        result_id = None
        if seg_files:  # Only save if segmentation was successful
            # Pass the output directory (which contains individual structure masks)
            # NOT a single structure file - results_db.py will derive structures_dir from this
            result_id = await save_segmentation_to_database(
                image_path=image_path,
                mask_path=str(output_dir),  # Directory containing all structure masks
                structures=structures_list,
                modality=modality,
                model_variant=model_variant,
                processing_time=processing_time,
                results_database=results_database  # Pass through database path
            )

        # Build response
        response = {
            "success": True,
            "structures": structures_list,
            "modality": modality,
            "model_variant": model_variant,
            "input_image": str(image_path),
            "output_directory": str(output_dir),
            "segmentation_files": seg_files,
            "num_files": len(seg_files),
            "processing_time": processing_time,
            "message": f"Successfully segmented {len(structures_list)} structures. Output saved to {output_dir}"
        }

        # Add database info if auto-saved
        if result_id:
            response["database_result_id"] = result_id
            response["auto_saved"] = True
            response["message"] += f" and automatically saved to database (ID: {result_id})"
        else:
            response["auto_saved"] = False

        # Add file details if we found segmentations
        if seg_files:
            response["primary_segmentation"] = seg_files[0]  # First file is usually the main result

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    except Exception as e:
        logger.error(f"✗ Segmentation failed: {e}", exc_info=True)
        raise


async def segment_structures_batch(
    image_paths: list[str],
    structures: list,
    results_database: str,
    modality: str = "mri",
    model_variant: str = "nano"
) -> list[TextContent]:
    """Execute SAT batch segmentation on multiple images using CLI interface.

    Args:
        image_paths: List of image file paths
        structures: Either single list for all images, or list of lists (one per image)
        results_database: Database path for storing segmentation results
        modality: Imaging modality (mri, ct, pet)
        model_variant: SAT model variant
    """

    logger.info(f"=" * 70)
    logger.info(f"SEGMENT_STRUCTURES_BATCH called")
    logger.info(f"  Images: {len(image_paths)} files")
    structures = _normalize_structures(structures)
    logger.info(f"  Structures: {structures}")
    logger.info(f"  Modality: {modality}")
    logger.info(f"  Model: {model_variant}")
    logger.info(f"=" * 70)

    try:
        # Resolve dataset identifiers to actual paths if needed
        logger.info(f"→ Resolving {len(image_paths)} image identifiers...")
        resolved_paths = [resolve_dataset_identifier(path) for path in image_paths]
        image_paths = resolved_paths
        logger.info(f"✓ Resolved all identifiers to paths")

        # Normalize modality to lowercase and validate
        modality = modality.lower()
        valid_modalities = ["mri", "ct", "pet"]
        if modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{modality}'. Must be one of: {valid_modalities}")
        logger.info(f"→ Normalized modality: {modality}")

        # Validate image paths
        for img_path in image_paths:
            img_path_obj = Path(img_path)
            if not img_path_obj.exists():
                logger.error(f"✗ Image not found: {img_path}")
                raise FileNotFoundError(f"Image not found: {img_path}")

        logger.info(f"✓ All {len(image_paths)} images found")

        # Check for existing results with structure-level granularity
        images_to_process = []
        images_with_missing_structures = {}  # img_path -> missing structures
        cached_results = {}

        if DATABASE_AVAILABLE:
            logger.info(f"→ Checking for existing results with structure-level caching...")

            for img_path in image_paths:
                patient_id, cardiac_phase = extract_patient_info_from_path(img_path)
                model_name = f"sat_{model_variant}"
                if cardiac_phase:
                    model_name += f"_{cardiac_phase.lower()}"

                try:
                    db = _get_db(results_database)

                    # Get all existing results for this patient/model
                    all_patient_results = await anyio.to_thread.run_sync(
                        lambda: [r for r in db.manifest["results"]
                                if r["patient_id"] == patient_id and r["model_name"] == model_name]
                    )

                    # Filter to results matching this specific source image (frame-level matching).
                    # Without this, a cache hit for patient041_frame01 (ED) would suppress
                    # segmentation of patient041_frame11 (ES), producing zero valid ES results.
                    img_basename = Path(img_path).name
                    existing_results = [
                        r for r in all_patient_results
                        if r.get("source_image") and Path(r["source_image"]).name == img_basename
                    ]

                    existing_structures = set()
                    for result in existing_results:
                        # Get structures from this result
                        result_structures = result.get("structures", [])
                        existing_structures.update(result_structures)

                    # Determine missing structures
                    requested_structures_set = set(structures)
                    missing_structures = requested_structures_set - existing_structures

                    if not missing_structures:
                        # All structures cached - full cache hit
                        logger.info(f"  ✓ Full cache hit for {Path(img_path).name} (all {len(structures)} structures)")
                        cached_results[img_path] = {
                            "success": True,
                            "segmentation_files": [],
                            "num_files": len(structures),
                            "processing_time": 0.0,
                            "database_result_id": existing_results[0].get("result_id", "") if existing_results else "",
                            "auto_saved": True,
                            "cached_result": True,
                            "cached_structures": list(existing_structures & requested_structures_set),
                            "structures_found": list(existing_structures & requested_structures_set),  # Add structures_found for agents
                            "message": f"All {len(structures)} structures found in cache"
                        }
                    elif existing_structures:
                        # Partial cache hit - some structures exist
                        cached_count = len(existing_structures & requested_structures_set)
                        logger.info(f"  ⚡ Partial cache for {Path(img_path).name}: {cached_count}/{len(structures)} cached, need {len(missing_structures)} more")
                        images_with_missing_structures[img_path] = list(missing_structures)
                        # Also track what we have cached
                        cached_results[img_path] = {
                            "success": True,
                            "partial_cache": True,
                            "cached_structures": list(existing_structures & requested_structures_set),
                            "missing_structures": list(missing_structures),
                            "num_cached": cached_count,
                            "num_missing": len(missing_structures)
                        }
                    else:
                        # No cache - need all structures
                        logger.info(f"  ⚠ Cache miss for {Path(img_path).name}, need all {len(structures)} structures")
                        images_to_process.append(img_path)

                except Exception as e:
                    logger.warning(f"Failed to check existing result for {Path(img_path).name}: {e}")
                    images_to_process.append(img_path)  # Process if check fails

            full_cache_hits = len([r for r in cached_results.values() if not r.get("partial_cache")])
            partial_cache_hits = len([r for r in cached_results.values() if r.get("partial_cache")])
            total_need_processing = len(images_to_process) + len(images_with_missing_structures)

            logger.info(f"  → {full_cache_hits} full cache hits, {partial_cache_hits} partial hits, {total_need_processing} need processing")
        else:
            images_to_process = image_paths  # Process all if no database

        # Process images: full cache misses + partial cache hits (with missing structures)
        all_images_needing_processing = images_to_process + list(images_with_missing_structures.keys())

        # Initialize result structure
        batch_result = {
            "success": True,
            "batch_size": len(image_paths),
            "processed_count": len(all_images_needing_processing),
            "cached_count": len(cached_results),
            "output_directory": "",
            "results": {}
        }
        
        # Add cached results to batch result
        for img_path, cached_data in cached_results.items():
            batch_result["results"][img_path] = cached_data

        if all_images_needing_processing:
            total_to_process = len(all_images_needing_processing)
            logger.info(f"→ Starting SAT batch inference for {total_to_process} images " +
                       f"({len(images_to_process)} full, {len(images_with_missing_structures)} partial)")

            # Determine output directory from results_database
            output_dir_for_sat = str(Path(results_database) / "outputs" / "sat_results")

            # Build per-image structure lists for partial cache hits
            # For full cache misses, use all requested structures
            per_image_structures = []
            for img_path in all_images_needing_processing:
                if img_path in images_with_missing_structures:
                    # Partial cache: only segment missing structures
                    per_image_structures.append(images_with_missing_structures[img_path])
                    logger.info(f"  → {Path(img_path).name}: segmenting {len(images_with_missing_structures[img_path])} missing structures")
                else:
                    # Full miss: segment all structures
                    per_image_structures.append(structures)
                    logger.info(f"  → {Path(img_path).name}: segmenting all {len(structures)} structures")

            # Use concurrency limiter to prevent multiple simultaneous GPU inferences
            async with _inference_limiter:
                import time
                start_time = time.time()

                result = await run_sat_inference_cli_batch_async(
                    image_paths=all_images_needing_processing,
                    structures=per_image_structures,  # Per-image structure lists
                    modality=modality,
                    model_variant=model_variant,
                    output_dir=output_dir_for_sat,
                    batchsize_3d=1  # Safe default for 24GB GPU
                )

                total_processing_time = time.time() - start_time

                # Merge SAT results with batch result
                batch_result["output_directory"] = result["output_directory"]
                batch_result["total_processing_time"] = total_processing_time

                # Add processed results to batch result
                for img_path, img_result in result["results"].items():
                    # If this was a partial cache hit, merge with existing cached data
                    if img_path in images_with_missing_structures:
                        existing_data = batch_result["results"].get(img_path, {})
                        # Merge cached structures with newly segmented ones
                        all_structures = existing_data.get("cached_structures", []) + images_with_missing_structures[img_path]
                        img_result["all_structures"] = all_structures
                        img_result["cached_structures"] = existing_data.get("cached_structures", [])
                        img_result["newly_segmented"] = images_with_missing_structures[img_path]
                        img_result["partial_cache_used"] = True
                        logger.info(f"  ⚡ {Path(img_path).name}: merged {len(existing_data.get('cached_structures', []))} cached + {len(images_with_missing_structures[img_path])} new")

                    batch_result["results"][img_path] = img_result
        
        else:
            logger.info(f"→ All results found in cache, no inference needed!")
            batch_result["total_processing_time"] = 0.0
            batch_result["output_directory"] = results_database
        
        logger.info(f"✓ Batch processing complete")
        logger.info(f"  Total images: {batch_result['batch_size']}")
        logger.info(f"  Processed: {batch_result['processed_count']}")
        logger.info(f"  Cached: {batch_result['cached_count']}")
        logger.info(f"  Output dir: {batch_result['output_directory']}")

        # Auto-save only newly processed results to database (cached ones are already saved)
        saved_results = {}
        newly_processed_count = 0
        
        if DATABASE_AVAILABLE and all_images_needing_processing:
            logger.info(f"→ Auto-saving {len(all_images_needing_processing)} newly processed results to database...")

            for img_path, img_result in batch_result["results"].items():
                # Only save results that were just processed (not cached)
                if not img_result.get("cached_result", False) and img_result["success"] and img_result.get("segmentation_files"):
                    try:
                        # Pass the output directory (which contains individual structure masks)
                        # NOT a single structure file - results_db.py will derive structures_dir from this
                        output_dir = Path(img_result.get("output_directory", ""))
                        structures_to_save = (
                            img_result.get("newly_segmented")
                            or img_result.get("structures_found")
                            or structures
                        )

                        result_id = await save_segmentation_to_database(
                            image_path=img_path,
                            mask_path=str(output_dir),  # Directory containing all structure masks
                            structures=structures_to_save,
                            modality=modality,
                            model_variant=model_variant,
                            processing_time=batch_result.get("total_processing_time", 0) / len(all_images_needing_processing)
                            if all_images_needing_processing else 0,
                            results_database=results_database  # Pass through database path
                        )
                        
                        if result_id:
                            saved_results[img_path] = result_id
                            img_result["database_result_id"] = result_id
                            img_result["auto_saved"] = True
                            newly_processed_count += 1
                            logger.info(f"  ✓ Saved {Path(img_path).name}: {result_id}")
                        else:
                            img_result["auto_saved"] = False
                    
                    except Exception as e:
                        logger.error(f"  ✗ Failed to save {Path(img_path).name}: {e}")
                        img_result["auto_saved"] = False
                        img_result["save_error"] = str(e)
                elif img_result.get("cached_result", False):
                    # Mark cached results as already saved
                    img_result["auto_saved"] = True
                else:
                    img_result["auto_saved"] = False

        # Add batch-level database info
        batch_result["database_saved_count"] = len(saved_results)
        batch_result["newly_processed_count"] = newly_processed_count
        batch_result["auto_save_enabled"] = DATABASE_AVAILABLE

        # Log per-image results
        for img_path, img_result in batch_result["results"].items():
            if img_result["success"]:
                status_parts = []
                if img_result.get("cached_result"):
                    status_parts.append("cached")
                if img_result.get("auto_saved"):
                    status_parts.append("saved")
                status = ", ".join(status_parts) if status_parts else "not saved"
                logger.info(f"  ✓ {Path(img_path).name}: {img_result['num_files']} files ({status})")
            else:
                logger.warning(f"  ✗ {Path(img_path).name}: {img_result.get('error', 'unknown error')}")

        # Create compact summary response (instead of full paths)
        # Full paths consume ~2KB per image × 80 images = 160KB
        # Summary reduces this to ~100 bytes per image = 8KB total
        summary = {
            "success": True,
            "batch_size": batch_result["batch_size"],
            "processed_count": batch_result["processed_count"],
            "cached_count": batch_result["cached_count"],
            "total_processing_time": batch_result.get("total_processing_time", 0),
            "output_directory": batch_result["output_directory"],
            "database_saved_count": batch_result.get("database_saved_count", 0),
            "auto_save_enabled": batch_result.get("auto_save_enabled", False),
            "images": []
        }
        
        # Add compact per-image summary (patient ID, status, DB ID only)
        for img_path, img_result in batch_result["results"].items():
            patient_id, cardiac_phase = extract_patient_info_from_path(img_path)
            image_summary = {
                "patient_id": patient_id,
                "phase": cardiac_phase,
                "success": img_result["success"],
                "num_structures": img_result.get("num_files", 0),
                "database_id": img_result.get("database_result_id", ""),
                "cached": img_result.get("cached_result", False),
                "saved": img_result.get("auto_saved", False)
            }
            
            # Include output directory so agents can find the actual files
            # e.g., .../seg_patient001_frame01/ contains all structure masks
            if img_result.get("output_directory"):
                image_summary["output_directory"] = img_result["output_directory"]
            
            # Include error details if failed
            if not img_result["success"]:
                image_summary["error"] = img_result.get("error", "unknown error")
            
            # Include structures list but NOT full paths
            # Agents can construct paths: {output_directory}/{structure}.nii.gz
            if img_result.get("structures_found"):
                image_summary["structures"] = img_result["structures_found"]
            
            summary["images"].append(image_summary)
        
        logger.info(f"→ Returning compact summary for {summary['batch_size']} image(s)")
        logger.info(f"  Summary size: ~{len(json.dumps(summary))} bytes (vs ~{len(json.dumps(batch_result))} bytes full)")
        return [TextContent(type="text", text=json.dumps(summary, indent=2))]

    except Exception as e:
        logger.error(f"✗ Batch segmentation failed: {e}", exc_info=True)
        raise


async def list_available_structures(
    category: str = None,
    search: str = None
) -> list[TextContent]:
    """List available anatomical structures for segmentation.

    Args:
        category: Optional category filter
        search: Optional search keyword
        category="all": Show complete catalog
    """

    # Normalize inputs (LLMs often pass invalid values for optional params)
    if search in ('', 'None', None):
        search = None
    if category in ('', 'None', None):
        category = None
    elif str(category).lower() == "all":
        category = None
        show_all = True
    else:
        show_all = False

    logger.info(f"LIST_AVAILABLE_STRUCTURES called (category={category}, search={search}, show_all={show_all})")

    try:
        if show_all:
            # Show complete organized catalog
            result = {
                "total_structures": len(search_structures("")),  # All structures
                "categories": {}
            }
            for cat in get_all_categories():
                cat_info = STRUCTURE_CATALOG[cat]
                result["categories"][cat] = {
                    "description": cat_info["description"],
                    "structures": cat_info["structures"],
                    "count": len(cat_info["structures"])
                }
            message = format_catalog_for_agents()
            result["formatted_catalog"] = message

        elif category:
            # Filter by category
            structures = get_structures_by_category(category)
            cat_info = STRUCTURE_CATALOG[category]
            result = {
                "category": category,
                "description": cat_info["description"],
                "structures": structures,
                "count": len(structures),
                "datasets": cat_info["datasets"]
            }

        elif search:
            # Search by keyword
            structures = search_structures(search)
            result = {
                "search_query": search,
                "matches": structures,
                "count": len(structures)
            }
            if len(structures) == 0:
                result["message"] = (
                    f"No structures found matching '{search}'. "
                    "Try a different keyword or use category='all' to see all options."
                )

        else:
            # Default: show categories overview
            categories_info = {}
            for cat in get_all_categories():
                cat_data = STRUCTURE_CATALOG[cat]
                categories_info[cat] = {
                    "description": cat_data["description"],
                    "count": len(cat_data["structures"]),
                    "examples": cat_data["structures"][:3]  # Show first 3 as examples
                }
            result = {
                "total_structures": len(search_structures("")),
                "total_categories": len(get_all_categories()),
                "categories": categories_info,
                "message": "Use 'category' parameter to see structures in a specific category, 'search' to find specific structures, or category='all' for complete catalog."
            }

        logger.info(f"→ Returning structure catalog info")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"✗ Structure catalog query failed: {e}", exc_info=True)
        raise


async def check_sat_status() -> list[TextContent]:
    """Check SAT model status and installation."""

    logger.info("Checking SAT installation status")

    try:
        # Run blocking check in thread pool to avoid blocking event loop
        status = await anyio.to_thread.run_sync(check_sat_installation)

        # Add helpful messages
        if not status["ready"]:
            messages = []
            if not status["sat_repo_exists"]:
                messages.append("SAT repository not found. Run: bash scripts/setup_sat.sh")
            if not status["inference_script_exists"]:
                messages.append("SAT inference.py not found in repository")
            if not status["checkpoint_dir_exists"]:
                messages.append("Checkpoint directory not found")
            elif not any(cp["complete"] for cp in status["checkpoints"].values()):
                messages.append("No complete checkpoint sets found. Run: bash scripts/setup_sat.sh")

            status["messages"] = messages
        else:
            # Find which checkpoints are ready
            ready_variants = [
                variant for variant, cp in status["checkpoints"].items()
                if cp["complete"]
            ]
            status["messages"] = [f"✓ SAT ready with {', '.join(ready_variants)} models"]

        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "ready": False,
                "error": str(e)
            }, indent=2)
        )]


async def async_main():
    """Run the MCP server asynchronously."""
    from mcp.server.stdio import stdio_server
    from mcp.server import InitializationOptions
    from mcp.types import ServerCapabilities, ToolsCapability

    # Check installation status on startup
    try:
        # Run in thread pool since this is a blocking I/O operation
        status = await anyio.to_thread.run_sync(check_sat_installation)
        logger.info(f"SAT Repository: {status['sat_repo_path']}")
        logger.info(f"Checkpoints: {status['checkpoint_dir']}")
        logger.info(f"Ready: {status['ready']}")

        if not status["ready"]:
            logger.warning("⚠️  SAT not fully configured. Run: bash scripts/setup_sat.sh")
        else:
            ready_models = [v for v, cp in status["checkpoints"].items() if cp["complete"]]
            logger.info(f"✓ Ready with models: {', '.join(ready_models)}")
    except Exception as e:
        logger.error(f"Failed to check SAT status: {e}")

    logger.info("Starting MCP stdio server")

    # Run MCP server via stdio asynchronously
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Stdio streams created, initializing server")

        # Initialize the server
        init_options = InitializationOptions(
            server_name="sat-segmentation",
            server_version="1.0.0",
            capabilities=ServerCapabilities(
                tools=ToolsCapability()
            ),
        )

        # Run the server with the streams
        logger.info("Starting server.run()")

        try:
            await server.run(
                read_stream,
                write_stream,
                init_options,
                raise_exceptions=True  # Raise exceptions for debugging
            )
        except Exception as e:
            logger.error(f"server.run() raised exception: {e}", exc_info=True)
            raise

        logger.info("server.run() completed")


def main():
    """Run the MCP server."""
    logger.info("SAT MCP Server Starting")

    # Use asyncio.run() - this is what official MCP servers use
    # stdio_server() uses anyio internally but works fine with asyncio.run()
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("MCP server interrupted")
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
