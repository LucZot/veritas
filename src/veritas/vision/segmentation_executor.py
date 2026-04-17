import asyncio
from pathlib import Path
from typing import Any, Dict, List

from veritas.mcp import MCPClientManager, load_mcp_config
from veritas.vision.datasets.dataset_tools import (
    list_dataset_patients,
    resolve_dataset_identifier,
)


def _invoke_tool(tool_fn, **kwargs):
    if hasattr(tool_fn, "invoke"):
        return tool_fn.invoke(kwargs)
    return tool_fn(**kwargs)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _get_sat_manager() -> MCPClientManager:
    mcp_manager = MCPClientManager()
    mcp_config = load_mcp_config()
    server_lookup = {server["name"]: server for server in mcp_config.get("servers", [])}
    if "sat" not in server_lookup:
        raise ValueError("SAT MCP server not found in mcp_servers.json")
    mcp_manager.register_server("sat", server_lookup["sat"])
    return mcp_manager


def _run_async(coro):
    return asyncio.run(coro)



async def _segment_batch(
    mcp_manager: MCPClientManager,
    image_paths: List[str],
    structures: List[str],
    results_database: str,
    modality: str,
    model_variant: str,
) -> Dict[str, Any]:
    return await mcp_manager.call_tool(
        "sat",
        "segment_structures_batch",
        {
            "image_paths": image_paths,
            "structures": structures,
            "results_database": results_database,
            "modality": modality,
            "model_variant": model_variant,
        },
    )


def segment_cohorts(
    dataset: str,
    case_label: str,
    control_label: str,
    observations: List[str],
    structures: List[str],
    results_database: str,
    modality: str = "mri",
    model_variant: str = "nano",
    chunk_size: int = 20,
) -> Dict[str, Any]:
    if not observations:
        raise ValueError("observations must be a non-empty list")
    if not structures:
        raise ValueError("structures must be a non-empty list")

    case_patients = _invoke_tool(
        list_dataset_patients, dataset=dataset, group=case_label
    )["patients"]
    control_patients = _invoke_tool(
        list_dataset_patients, dataset=dataset, group=control_label
    )["patients"]

    patient_ids = [p["patient_id"] for p in case_patients + control_patients]
    if not patient_ids:
        raise ValueError("No patients found for the requested cohorts")

    identifiers = [
        f"{dataset}:{patient_id}:{observation}"
        for patient_id in patient_ids
        for observation in observations
    ]

    results_db_path = _resolve_path(results_database)
    results_db_path.mkdir(parents=True, exist_ok=True)

    mcp_manager = _get_sat_manager()
    chunk_size = max(1, int(chunk_size))

    summary: Dict[str, Any] = {
        "success": True,
        "dataset": dataset,
        "case_label": case_label,
        "control_label": control_label,
        "case_count": len(case_patients),
        "control_count": len(control_patients),
        "total_identifiers": len(identifiers),
        "observations": observations,
        "structures": structures,
        "results_database": str(results_db_path),
        "chunks": [],
        "batch_size": 0,
        "processed_count": 0,
        "cached_count": 0,
        "images": [],
        "errors": [],
    }

    for idx in range(0, len(identifiers), chunk_size):
        chunk = identifiers[idx: idx + chunk_size]
        result = _run_async(
            _segment_batch(
                mcp_manager,
                chunk,
                structures,
                str(results_db_path),
                modality,
                model_variant,
            )
        )
        error = result.get("error")
        summary["chunks"].append(
            {
                "chunk_index": idx // chunk_size,
                "batch_size": result.get("batch_size", len(chunk)),
                "processed_count": result.get("processed_count", 0),
                "cached_count": result.get("cached_count", 0),
                "success": result.get("success", False),
                "error": error,
            }
        )
        if error:
            summary["errors"].append(
                {
                    "chunk_index": idx // chunk_size,
                    "error": error,
                }
            )
        summary["batch_size"] += result.get("batch_size", len(chunk))
        summary["processed_count"] += result.get("processed_count", 0)
        summary["cached_count"] += result.get("cached_count", 0)
        summary["success"] = summary["success"] and result.get("success", False)
        summary["images"].extend(result.get("images", []))

    return summary


def segment_identifiers(
    identifiers: List[str],
    structures: List[str],
    results_database: str,
    modality: str = "mri",
    model_variant: str = "nano",
    chunk_size: int = 20,
) -> Dict[str, Any]:
    if not identifiers:
        raise ValueError("identifiers must be a non-empty list")
    if not structures:
        raise ValueError("structures must be a non-empty list")

    results_db_path = _resolve_path(results_database)
    results_db_path.mkdir(parents=True, exist_ok=True)

    mcp_manager = _get_sat_manager()
    chunk_size = max(1, int(chunk_size))

    # Resolve identifiers to absolute paths before sending to SAT server,
    # so the server doesn't need dataset-specific path configuration
    resolved = []
    for ident in identifiers:
        if ":" in ident and not Path(ident).is_absolute():
            info = _invoke_tool(resolve_dataset_identifier, identifier=ident)
            resolved.append(info["file_path"])
        else:
            resolved.append(ident)

    summary: Dict[str, Any] = {
        "success": True,
        "total_identifiers": len(resolved),
        "structures": structures,
        "results_database": str(results_db_path),
        "chunks": [],
        "batch_size": 0,
        "processed_count": 0,
        "cached_count": 0,
        "images": [],
        "errors": [],
    }

    for idx in range(0, len(resolved), chunk_size):
        chunk = resolved[idx: idx + chunk_size]
        result = _run_async(
            _segment_batch(
                mcp_manager,
                chunk,
                structures,
                str(results_db_path),
                modality,
                model_variant,
            )
        )
        error = result.get("error")
        summary["chunks"].append(
            {
                "chunk_index": idx // chunk_size,
                "batch_size": result.get("batch_size", len(chunk)),
                "processed_count": result.get("processed_count", 0),
                "cached_count": result.get("cached_count", 0),
                "success": result.get("success", False),
                "error": error,
            }
        )
        if error:
            summary["errors"].append(
                {
                    "chunk_index": idx // chunk_size,
                    "error": error,
                }
            )
        summary["batch_size"] += result.get("batch_size", len(chunk))
        summary["processed_count"] += result.get("processed_count", 0)
        summary["cached_count"] += result.get("cached_count", 0)
        summary["success"] = summary["success"] and result.get("success", False)
        summary["images"].extend(result.get("images", []))

    return summary
