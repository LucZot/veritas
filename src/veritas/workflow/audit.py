"""Workflow audit utilities: validation reports, correction reports, and failure taxonomy."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


FAILURE_CODES = {
    "P1_SCHEMA_INVALID",
    "P1_SEMANTIC_INVALID_OBSERVATIONS",
    "P2A_REQUEST_MISSING_KEYS",
    "P2A_EMPTY_IDENTIFIERS",
    "P2A_OBSERVATION_MISMATCH",
    "P2B_RESULTS_MISSING",
    "P2B_RESULTS_SCHEMA_INVALID",
    "P2B_SYNTHETIC_DATA",
    "P2B_OFF_CONTRACT_DATA_LOADING",
    "P2B_INVALID_SAMPLE_SIZE",
    "P3_VERDICT_SCHEMA_INVALID",
    "INFRA_OLLAMA_STREAM",
    "INFRA_TIMEOUT",
    "INFRA_IMPORT_RUNTIME",
    "INFRA_CONNECTION_ERROR",
    "INFRA_UNKNOWN",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if default is None:
        default = {}
    if not path.exists():
        return dict(default)
    try:
        with open(path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return dict(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_validation_report(
    *,
    save_dir: Path,
    phase: str,
    status: str,
    checks: Optional[Dict[str, Any]] = None,
    errors: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
    failure_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a normalized per-phase validation report."""
    payload: Dict[str, Any] = {
        "phase": phase,
        "status": status,
        "timestamp_utc": _utc_now(),
        "checks": checks or {},
        "errors": errors or [],
        "warnings": warnings or [],
        "failure_code": failure_code,
    }
    if details:
        payload["details"] = details
    _write_json(save_dir / "validation_report.json", payload)


def append_correction_report(
    *,
    save_dir: Path,
    phase: str,
    attempted: bool,
    success: bool,
    reason: str,
    error_before: Optional[str] = None,
    error_after: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a correction attempt entry to correction_report.json."""
    path = save_dir / "correction_report.json"
    report = _read_json(path, default={"phase": phase, "attempts": []})
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        attempts = []

    entry: Dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "attempted": attempted,
        "success": success,
        "reason": reason,
        "error_before": error_before,
        "error_after": error_after,
    }
    if details:
        entry["details"] = details
    attempts.append(entry)
    report["phase"] = phase
    report["attempts"] = attempts
    _write_json(path, report)


def update_workflow_audit(
    *,
    config_file: Path,
    phase: str,
    status: str,
    failure_code: Optional[str] = None,
    failure_message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    overwrite_failure: bool = False,
) -> None:
    """Update workflow_config.json with normalized workflow_audit metadata."""
    if not config_file.exists():
        return

    try:
        with open(config_file) as f:
            config = json.load(f)
    except Exception:
        return

    audit = config.get("workflow_audit")
    if not isinstance(audit, dict):
        audit = {}

    phase_status = audit.get("phase_status")
    if not isinstance(phase_status, dict):
        phase_status = {}
    phase_status[phase] = {
        "status": status,
        "timestamp_utc": _utc_now(),
    }
    if details:
        phase_status[phase]["details"] = details
    audit["phase_status"] = phase_status

    if status == "passed" and audit.get("failure_stage") == phase:
        for key in ("failure_code", "failure_stage", "failure_message", "failure_timestamp_utc"):
            audit.pop(key, None)

    if failure_code:
        should_write_failure = overwrite_failure or not audit.get("failure_code")
        if should_write_failure:
            audit["failure_code"] = failure_code
            audit["failure_stage"] = phase
            audit["failure_message"] = failure_message
            audit["failure_timestamp_utc"] = _utc_now()

    config["workflow_audit"] = audit
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        return


def map_transient_reason_to_failure_code(reason: Optional[str]) -> str:
    if reason == "ollama_stream":
        return "INFRA_OLLAMA_STREAM"
    if reason == "transport_timeout":
        return "INFRA_TIMEOUT"
    if reason in {"import_error", "runtime_library_error"}:
        return "INFRA_IMPORT_RUNTIME"
    if reason == "connection_error":
        return "INFRA_CONNECTION_ERROR"
    return "INFRA_UNKNOWN"


def classify_failure_code(phase: str, error_message: str) -> str:
    """Map a phase exception message to a stable failure taxonomy code."""
    msg = (error_message or "").strip()
    phase_norm = phase.lower().strip()

    if phase_norm == "phase1":
        if "observations must be imaging observations/timepoints only" in msg.lower():
            return "P1_SEMANTIC_INVALID_OBSERVATIONS"
        if "observations" in msg.lower() and "available observations" in msg.lower():
            return "P1_SEMANTIC_INVALID_OBSERVATIONS"
        return "P1_SCHEMA_INVALID"

    if phase_norm == "phase2a":
        if "missing required keys" in msg.lower():
            return "P2A_REQUEST_MISSING_KEYS"
        if "identifiers list is empty" in msg.lower():
            return "P2A_EMPTY_IDENTIFIERS"
        if "observations not in plan" in msg.lower() or "malformed identifiers" in msg.lower():
            return "P2A_OBSERVATION_MISMATCH"
        if "plan missing 'observations' field" in msg.lower():
            return "P2A_OBSERVATION_MISMATCH"
        return "P2A_REQUEST_MISSING_KEYS"

    if phase_norm == "phase2b":
        if "synthetic/mock data usage detected" in msg.lower():
            return "P2B_SYNTHETIC_DATA"
        if "off-contract data loading detected" in msg.lower():
            return "P2B_OFF_CONTRACT_DATA_LOADING"
        if "sample capping/subsampling detected" in msg.lower():
            return "P2B_INVALID_SAMPLE_SIZE"
        if "statistical results not found" in msg.lower():
            return "P2B_RESULTS_MISSING"
        if "missing required keys" in msg.lower():
            return "P2B_RESULTS_SCHEMA_INVALID"
        if "n_total must be a positive integer" in msg.lower():
            return "P2B_INVALID_SAMPLE_SIZE"
        return "P2B_RESULTS_SCHEMA_INVALID"

    if phase_norm == "phase3":
        return "P3_VERDICT_SCHEMA_INVALID"

    return "INFRA_UNKNOWN"


def detect_off_contract_data_loading(code_dir: Path, code_files: Optional[list[Path]] = None) -> Dict[str, Any]:
    """Detect primary raw filesystem/dataframe loading patterns in code artifacts."""
    if not code_dir.exists():
        return {"hard_hits": [], "soft_hits": []}

    filesystem_patterns = [
        ("filesystem_scan", re.compile(r"\bos\.listdir\s*\(")),
        ("filesystem_glob", re.compile(r"\bglob\s*\(")),
        ("path_glob", re.compile(r"\bPath\s*\(.*\)\.glob\s*\(")),
    ]
    tabular_patterns = [
        ("csv_loading", re.compile(r"\bread_csv\s*\(")),
        ("excel_loading", re.compile(r"\bread_excel\s*\(")),
        ("parquet_loading", re.compile(r"\bread_parquet\s*\(")),
        ("pickle_loading", re.compile(r"\bread_pickle\s*\(")),
    ]
    raw_image_patterns = [
        ("nib_load", re.compile(r"\bnib(?:abel)?\.load\s*\(")),
        ("sitk_read", re.compile(r"\b(?:sitk|SimpleITK)\.ReadImage\s*\(")),
        ("dicom_read", re.compile(r"\bpydicom\.dcmread\s*\(")),
        ("numpy_load", re.compile(r"\bnp\.load\s*\(")),
    ]
    dataset_indicator = re.compile(
        r"dataset_path|DATASET_PATH|raw(?:_|/)data|/images?/|/labels?/|"
        r"\.nii(?:\.gz)?|\.(?:dcm|mha|nrrd)\b",
        re.IGNORECASE,
    )
    workspace_indicator = re.compile(
        r"workspace|plots?/|data/statistical_results\.json|phase2b_run_manifest\.json|segmentation_request\.json",
        re.IGNORECASE,
    )

    if code_files is None:
        files_to_scan = sorted(code_dir.rglob("*.py"))
    else:
        code_root = code_dir.resolve()
        files_to_scan = []
        for candidate in code_files:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(code_root)
            except Exception:
                continue
            if resolved.exists() and resolved.suffix == ".py":
                files_to_scan.append(resolved)
        files_to_scan = sorted(set(files_to_scan))

    hard_hits = []
    soft_hits = []
    for py_file in files_to_scan:
        try:
            lines = py_file.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        rel = str(py_file)
        for lineno, line in enumerate(lines, start=1):
            lowered = line.lower()
            window = " ".join(
                l.lower()
                for l in lines[max(0, lineno - 2): min(len(lines), lineno + 2)]
            )
            window_has_dataset_hint = bool(dataset_indicator.search(window))
            window_is_workspace_local = bool(workspace_indicator.search(window))

            for rule_name, pattern in filesystem_patterns:
                if pattern.search(line):
                    hit = {
                        "file": rel,
                        "line": lineno,
                        "rule": rule_name,
                        "text": lowered.strip()[:220],
                    }
                    if window_has_dataset_hint and not window_is_workspace_local:
                        hard_hits.append(hit)
                    else:
                        soft_hits.append(hit)
            for rule_name, pattern in tabular_patterns:
                if pattern.search(line):
                    hit = {
                        "file": rel,
                        "line": lineno,
                        "rule": rule_name,
                        "text": lowered.strip()[:220],
                    }
                    if window_has_dataset_hint and not window_is_workspace_local:
                        hard_hits.append(hit)
                    else:
                        soft_hits.append(hit)
            for rule_name, pattern in raw_image_patterns:
                if pattern.search(line):
                    hit = {
                        "file": rel,
                        "line": lineno,
                        "rule": rule_name,
                        "text": lowered.strip()[:220],
                    }
                    if window_has_dataset_hint and not window_is_workspace_local:
                        hard_hits.append(hit)
                    else:
                        soft_hits.append(hit)

    return {"hard_hits": hard_hits, "soft_hits": soft_hits}
