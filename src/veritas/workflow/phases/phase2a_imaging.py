#!/usr/bin/env python3
"""Phase 2A: Segmentation Request

Agent writes code to enumerate patients and build segmentation_request.json.
Script then executes the segmentation via MCP.
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EXECUTION_MODE", "langgraph")
os.environ.setdefault("OLLAMA_CONTEXT_LENGTH", os.environ.get("VERITAS_CONTEXT_LENGTH", "16384"))

repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from veritas import run_meeting
from veritas.prompts import (
    PHASE_AWARE_CRITIC,
    CODING_MEDICAL_IMAGING_SPECIALIST_CODE_OUTPUT,
    create_agent_with_model,
)
from veritas.utils import archive_meeting_artifacts, build_correction_summary_sources, load_summaries
from veritas.verbosity import get_prompt_verbosity_config
from veritas.vision.segmentation_executor import segment_identifiers
from veritas.prompt_templates import build_phase2a_agenda
from veritas.workflow.audit import (
    append_correction_report,
    classify_failure_code,
    update_workflow_audit,
    write_validation_report,
)


def normalize_group_values(values) -> list[str]:
    """Flatten nested group values and coerce them to non-empty strings."""
    flattened: list[str] = []

    def _visit(node) -> None:
        if node is None:
            return
        if isinstance(node, (list, tuple, set)):
            for item in node:
                _visit(item)
            return
        token = str(node).strip()
        if token:
            flattened.append(token)

    _visit(values)
    return flattened


def normalize_restrict_to(value) -> dict:
    if not isinstance(value, dict):
        return {}
    normalized = dict(value)
    if "group" in normalized:
        groups = normalize_group_values(normalized.get("group"))
        if not groups:
            normalized.pop("group", None)
        elif len(groups) == 1:
            normalized["group"] = groups[0]
        else:
            normalized["group"] = groups
    return normalized


@dataclass
class PhaseConfig:
    """Configuration for phase execution."""
    output_path: str = "outputs/workflow"
    imaging_execution_model: str = "gpt-oss:20b"
    critic_model: str = "qwen3:8b"
    temperature: float = 0.2
    top_p: Optional[float] = None
    # Per-role temperature overrides (None = use global temperature)
    imaging_execution_temperature: Optional[float] = None
    critic_temperature: Optional[float] = None
    # Per-role top_p overrides (None = use global top_p)
    imaging_execution_top_p: Optional[float] = None
    critic_top_p: Optional[float] = None
    prompt_verbosity: str = "standard"
    dataset_path: str = ""
    dataset_metadata: Optional[dict] = None
    sat_cache_database: Optional[str] = None
    use_ground_truth: bool = False
    enable_critic: bool = True
    sat_model_variant: Optional[str] = None  # Override agent's model choice (e.g. "pro")


def load_config() -> PhaseConfig:
    """Load configuration from config file or environment variables."""
    config_file = os.environ.get("WORKFLOW_CONFIG_FILE")

    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            data = json.load(f)

        models = data.get("models") or {}
        temperatures = data.get("temperatures") or {}
        top_ps = data.get("top_ps") or {}

        return PhaseConfig(
            output_path=data.get("output_path", PhaseConfig.output_path),
            imaging_execution_model=models.get("imaging_execution", data.get("imaging_execution_model", PhaseConfig.imaging_execution_model)),
            critic_model=models.get("critic", data.get("critic_model", PhaseConfig.critic_model)),
            temperature=data.get("temperature", PhaseConfig.temperature),
            top_p=data.get("top_p"),
            imaging_execution_temperature=temperatures.get("imaging_execution"),
            critic_temperature=temperatures.get("critic"),
            imaging_execution_top_p=top_ps.get("imaging_execution"),
            critic_top_p=top_ps.get("critic"),
            prompt_verbosity=data.get("prompt_verbosity", PhaseConfig.prompt_verbosity),
            dataset_path=data.get("dataset_path", ""),
            dataset_metadata=data.get("dataset"),
            sat_cache_database=data.get("sat_cache_database"),
            use_ground_truth=data.get("use_ground_truth", False),
            enable_critic=data.get("enable_critic", True),
            sat_model_variant=data.get("sat_model_variant"),
        )

    # Fall back to environment variables
    metadata_str = os.environ.get("DATASET_METADATA")
    dataset_metadata = json.loads(metadata_str) if metadata_str else None

    return PhaseConfig(
        output_path=os.environ.get("WORKFLOW_OUTPUT_PATH", PhaseConfig.output_path),
        imaging_execution_model=os.environ.get("IMAGING_EXECUTION_MODEL", PhaseConfig.imaging_execution_model),
        critic_model=os.environ.get("CRITIC_MODEL", PhaseConfig.critic_model),
        temperature=float(os.environ.get("WORKFLOW_TEMPERATURE", str(PhaseConfig.temperature))),
        prompt_verbosity=os.environ.get("WORKFLOW_PROMPT_VERBOSITY", PhaseConfig.prompt_verbosity),
        dataset_path=os.environ.get("DATASET_PATH", ""),
        dataset_metadata=dataset_metadata,
    )


def main():
    """Run Phase 2A: Segmentation request preparation."""
    print("\n" + "=" * 70)
    print("PHASE 2A: SEGMENTATION REQUEST")
    print("=" * 70 + "\n")

    config = load_config()
    config_file = os.environ.get("WORKFLOW_CONFIG_FILE")
    if not config_file or not Path(config_file).exists():
        raise RuntimeError("WORKFLOW_CONFIG_FILE not found")
    config_path_obj = Path(config_file)
    update_workflow_audit(
        config_file=config_path_obj,
        phase="phase2a",
        status="running",
        details={"script": "phase2a_imaging.py"},
    )

    # Predeclare for failure reporting
    save_dir: Optional[Path] = None
    correction_attempted = False
    correction_success = None

    try:
        # Resolve output path
        output_path = Path(config.output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

        save_dir = (output_path / "phase2a_imaging_analysis").absolute()
        save_dir.mkdir(parents=True, exist_ok=True)

        workspace_dir = save_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CODE_EXEC_WORKSPACE_DIR"] = str(workspace_dir)

        # Set DATASET_PATH for dataset loader functions in code execution
        if config.dataset_path:
            dataset_path = Path(config.dataset_path)
            if not dataset_path.is_absolute():
                dataset_path = repo_root / dataset_path
            os.environ["DATASET_PATH"] = str(dataset_path)

        # Determine results database path
        if config.sat_cache_database:
            results_db = Path(config.sat_cache_database)
            if not results_db.is_absolute():
                results_db = repo_root / results_db
            results_db.mkdir(parents=True, exist_ok=True)
        else:
            results_db = save_dir / "results_database"
            results_db.mkdir(parents=True, exist_ok=True)

        # Load Phase 1 summary
        phase1_json = (output_path / "phase1_hypothesis_formulation" / "discussion.json").absolute()
        summaries = ()
        if phase1_json.exists():
            summaries = load_summaries([phase1_json])

        # Load plan
        with open(config_file) as f:
            full_config = json.load(f)
        plan = full_config.get("plan", {})

        if not plan:
            raise ValueError("No plan found in config - Phase 1 must run first")

        # Extract plan values
        groups = normalize_group_values(plan.get("groups", []))
        cohort_mode = str(plan.get("cohort_mode", "")).strip().lower()
        structures = plan.get("structures", [])
        observations = plan.get("observations", [])
        analysis_type = plan.get("analysis_type") or "group_difference"
        restrict_to = normalize_restrict_to(plan.get("restrict_to") or {})
        dataset_name = config.dataset_metadata.get("name", "dataset") if config.dataset_metadata else "dataset"

        group_tokens = [str(g).strip().lower() for g in groups]
        available_groups = normalize_group_values(
            config.dataset_metadata.get("available_groups", [])
            if config.dataset_metadata else []
        )
        all_groups_selected = False
        restrict_to_for_prompt = dict(restrict_to)
        if not groups or group_tokens == ["all"] or cohort_mode == "all":
            # If restrict_to specifies a group, use only that group instead of all
            if restrict_to and "group" in restrict_to:
                groups = normalize_group_values([restrict_to["group"]])
                if len(groups) > 1:
                    restrict_to_for_prompt.pop("group", None)
            else:
                if not available_groups:
                    raise ValueError("Plan requires groups or dataset metadata available_groups for ALL cohort")
                groups = list(available_groups)
                all_groups_selected = True

        if not groups:
            raise ValueError("Plan missing 'groups' field")

        # Metadata-only analyses (e.g. pure metadata correlations, survival by metadata group)
        # have no structures to segment — skip Phase 2A entirely.
        if not structures:
            print("  ℹ️  Metadata-only analysis: no structures or observations in plan.")
            print("  ℹ️  Skipping segmentation — writing stub segmentation_execution.json.")
            stub = {
                "success": True,
                "skipped": True,
                "reason": "metadata-only analysis (no structures/observations in plan)",
                "total_identifiers": 0,
                "structures": [],
                "results_database": str(results_db),
                "cached_count": 0,
                "processed_count": 0,
            }
            execution_path = save_dir / "segmentation_execution.json"
            execution_path.write_text(json.dumps(stub, indent=2))
            summary_path = save_dir / "summary.md"
            summary_path.write_text("### Execution Summary\n\n- Skipped: metadata-only analysis\n")
            write_validation_report(
                save_dir=save_dir,
                phase="phase2a",
                status="passed",
                checks={"metadata_only_skip": True, "correction_attempted": False},
                warnings=["phase2a_skipped_metadata_only"],
            )
            update_workflow_audit(
                config_file=config_path_obj,
                phase="phase2a",
                status="passed",
                details={"metadata_only_skip": True},
            )
            print(f"\n✓ Phase 2A skipped (metadata-only). Stub written to {execution_path}")
            return

        if "structures" not in plan:
            raise ValueError("Plan missing 'structures' field")
        if not observations:
            raise ValueError("Plan missing 'observations' field")

        # For metadata-group comparisons (groups not in available_groups), the group-count
        # check is deferred to Phase 2B where the metadata split happens.
        available_groups_set = set(str(g).strip() for g in available_groups)
        api_groups = [g for g in groups if str(g).strip() in available_groups_set]
        if analysis_type == "group_difference" and len(groups) < 2 and len(api_groups) == len(groups):
            raise ValueError("Group-difference analysis requires at least two groups")

        available_observations = (
            config.dataset_metadata.get("available_observations", [])
            if config.dataset_metadata else []
        )
        if available_observations:
            available_lower = {str(obs).strip().lower() for obs in available_observations}
            invalid_obs = [obs for obs in observations if str(obs).strip().lower() not in available_lower]
            if invalid_obs:
                raise ValueError(
                    f"Plan observations {invalid_obs} not in dataset available_observations {available_observations}."
                )

        # Get modality from dataset metadata
        modality = "mri"  # default
        if config.dataset_metadata:
            modality_str = config.dataset_metadata.get("modality", "").lower()
            if "mri" in modality_str or "mr" in modality_str:
                modality = "mri"
            elif "ct" in modality_str:
                modality = "ct"
            elif "pet" in modality_str:
                modality = "pet"
            else:
                print(f"  ⚠ Warning: Could not determine modality from '{config.dataset_metadata.get('modality', '')}', defaulting to 'mri'")

        # Create agents
        imaging_specialist = create_agent_with_model(
            CODING_MEDICAL_IMAGING_SPECIALIST_CODE_OUTPUT,
            config.imaging_execution_model,
            config.imaging_execution_temperature,
            config.imaging_execution_top_p,
        )
        critic = create_agent_with_model(
            PHASE_AWARE_CRITIC,
            config.critic_model,
            config.critic_temperature,
            config.critic_top_p,
        )

        # Determine which groups are direct API groups vs metadata-based
        available_groups_set = set(available_groups)
        metadata_groups = [g for g in groups if g not in available_groups_set]
        grouping_field = plan.get("grouping_field")
        group_spec = plan.get("group_spec") if isinstance(plan.get("group_spec"), dict) else None
        if not grouping_field and group_spec and str(group_spec.get("type", "")).lower() == "metadata":
            grouping_field = group_spec.get("field")
        metadata_fields = (
            config.dataset_metadata.get("patient_metadata_fields", [])
            if config.dataset_metadata else []
        )

        agenda = build_phase2a_agenda(
            plan=plan,
            dataset_name=dataset_name,
            groups=groups,
            structures=structures,
            observations=observations,
            results_db=str(results_db),
            modality=modality,
            available_groups=available_groups,
            metadata_fields=metadata_fields,
            all_groups_selected=all_groups_selected,
            metadata_groups=metadata_groups,
            cohort_mode=cohort_mode,
            restrict_to=restrict_to_for_prompt,
        )

        print(f"Running agent meeting with:")
        print(f"  • Imaging Specialist: {config.imaging_execution_model}")
        print(f"  • Critic: {config.critic_model}")
        print(f"  • Temperature: {config.temperature}")
        print(f"  • Verbosity: {config.prompt_verbosity}")
        print()

        run_meeting(
            meeting_type="individual",
            team_member=imaging_specialist,
            critic=critic,
            agenda=agenda,
            summaries=summaries,
            save_dir=save_dir,
            num_rounds=1,
            temperature=config.temperature,
            top_p=config.top_p,
            prompt_verbosity=config.prompt_verbosity,
            enable_critic=config.enable_critic,
            workflow_instruction="Phase 2A: Segmentation Request - Build request JSON",
        )

        print(f"\n✓ Phase 2A agent complete. Results saved to {save_dir}")
        print(f"  • Discussion: {save_dir / 'discussion.json'}")
        print(f"  • Summary: {save_dir / 'summary.md'}")

        # Load and validate segmentation request (single correction pass on failure)
        request = None
        last_request_error = None
        for attempt in range(2):
            try:
                request = load_segmentation_request(workspace_dir)
                break
            except (FileNotFoundError, ValueError) as error:
                last_request_error = error
                if attempt == 1:
                    correction_success = False
                    append_correction_report(
                        save_dir=save_dir,
                        phase="phase2a",
                        attempted=correction_attempted,
                        success=False,
                        reason="segmentation_request_still_invalid_after_correction",
                        error_before=str(error),
                        error_after=str(error),
                    )
                    raise
                print(f"\n  ⚠ Segmentation request validation failed: {error}")
                print("  → Requesting one correction pass from Imaging Specialist...")
                correction_attempted = True
                _request_segmentation_request_correction(
                    save_dir=save_dir,
                    phase1_json=phase1_json,
                    imaging_specialist=imaging_specialist,
                    critic=critic,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    prompt_verbosity=config.prompt_verbosity,
                    enable_critic=config.enable_critic,
                    error_message=str(error),
                    grouping_field=str(grouping_field) if grouping_field is not None else None,
                    metadata_groups=[str(value) for value in metadata_groups],
                    available_groups=[str(value) for value in available_groups],
                    restrict_to=restrict_to_for_prompt if isinstance(restrict_to_for_prompt, dict) else {},
                    correction_index=1,
                )

        if request is None and last_request_error is not None:
            raise last_request_error

        if correction_attempted:
            correction_success = True
            append_correction_report(
                save_dir=save_dir,
                phase="phase2a",
                attempted=True,
                success=True,
                reason="segmentation_request_correction_succeeded",
                error_before=str(last_request_error) if last_request_error else None,
            )

        identifiers = request["identifiers"]
        structures = request["structures"]
        results_database = request["results_database"]
        modality = request.get("modality", "mri")
        model_variant = request.get("model_variant", "nano")
        if config.sat_model_variant:
            model_variant = config.sat_model_variant
        chunk_size = request.get("chunk_size", 20)
        validate_request_observations(request, observations)

        print(f"\n📋 Segmentation Request:")
        print(f"  • Identifiers: {len(identifiers)} samples")
        print(f"  • Structures: {', '.join(structures)}")
        print(f"  • Database: {results_database}")
        print(f"  • Modality: {modality}")
        print(f"  • Model: {model_variant}")
        print()

        # Execute segmentation (or populate from ground truth)
        if config.use_ground_truth:
            from veritas.vision.gt_populator import populate_gt_results
            gt_results_database = str(Path(results_database) / "ground_truth")
            print(f"🔬 Populating ground truth masks → {gt_results_database}")
            dataset_name = (config.dataset_metadata or {}).get("name")
            execution_summary = populate_gt_results(
                identifiers=identifiers,
                structures=structures,
                results_database=gt_results_database,
                dataset_path=config.dataset_path,
                dataset_name=dataset_name,
            )
        else:
            print("🔬 Executing segmentation...")
            execution_summary = segment_identifiers(
                identifiers=identifiers,
                structures=structures,
                results_database=results_database,
                modality=modality,
                model_variant=model_variant,
                chunk_size=chunk_size,
            )

        execution_path = save_dir / "segmentation_execution.json"
        execution_path.write_text(json.dumps(execution_summary, indent=2))

        # Write summary
        write_execution_summary(
            save_dir=save_dir,
            execution_summary=execution_summary,
            request=request,
        )

        if execution_summary.get("errors"):
            print(f"  • Errors: {len(execution_summary['errors'])} (see segmentation_execution.json)")

        if not execution_summary.get("success", False):
            raise RuntimeError(
                "Segmentation incomplete. Check segmentation_execution.json for details."
            )

        write_validation_report(
            save_dir=save_dir,
            phase="phase2a",
            status="passed",
            checks={
                "segmentation_request_present": True,
                "segmentation_success": bool(execution_summary.get("success", False)),
                "identifier_count": len(request.get("identifiers", [])),
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase2a",
            status="passed",
            details={
                "identifier_count": len(request.get("identifiers", [])),
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )

        print(f"\n✓ Phase 2A complete.")
        print(f"  • Segmentation execution: {execution_path}")
        print(f"  • Processed: {execution_summary.get('processed_count', 0)} identifiers")
        print(f"  • Cached: {execution_summary.get('cached_count', 0)} identifiers")
    except Exception as e:
        failure_code = classify_failure_code("phase2a", str(e))
        if save_dir:
            write_validation_report(
                save_dir=save_dir,
                phase="phase2a",
                status="failed",
                checks={
                    "correction_attempted": correction_attempted,
                    "correction_success": correction_success,
                },
                errors=[str(e)],
                failure_code=failure_code,
            )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase2a",
            status="failed",
            failure_code=failure_code,
            failure_message=str(e),
            details={
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        raise


def load_segmentation_request(workspace_dir: Path) -> dict:
    """Load and validate segmentation request.

    Checks workspace root first, then data/ subdirectory as fallback.
    """
    # Primary: workspace root (expected location)
    request_path = workspace_dir / "segmentation_request.json"

    if not request_path.exists():
        # Fallback: agents sometimes write to data/ by mistake
        data_candidate = workspace_dir / "data" / "segmentation_request.json"
        if data_candidate.exists():
            print(f"  WARNING: segmentation_request.json found in data/ instead of workspace root — using it")
            request_path = data_candidate

    if not request_path.exists():
        raise FileNotFoundError(
            f"Expected file at: {workspace_dir}/segmentation_request.json\n"
            "Agent must write segmentation_request.json to the workspace root"
        )

    with open(request_path) as f:
        request = json.load(f)

    # Validate required keys
    required = ["identifiers", "structures", "results_database"]
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(
            f"Segmentation request missing required keys: {', '.join(missing)}"
        )

    if not request["identifiers"]:
        raise ValueError("Segmentation request identifiers list is empty")
    if not request["structures"]:
        raise ValueError("Segmentation request structures list is empty")

    return request


def _request_segmentation_request_correction(
    correction_index: int,
    *,
    save_dir: Path,
    phase1_json: Path,
    imaging_specialist,
    critic,
    temperature: float,
    top_p: Optional[float] = None,
    prompt_verbosity: str,
    enable_critic: bool,
    error_message: str,
    grouping_field: Optional[str] = None,
    metadata_groups: Optional[list[str]] = None,
    available_groups: Optional[list[str]] = None,
    restrict_to: Optional[dict] = None,
) -> None:
    metadata_groups = metadata_groups or []
    available_groups = available_groups or []
    restrict_to = restrict_to or {}
    metadata_hint = ""
    if metadata_groups:
        metadata_hint = (
            "5) For metadata-based groups, inspect real metadata values first and map plan labels to exact canonical values\n"
            f"   - Planned metadata groups: {metadata_groups}\n"
            f"   - grouping_field: {grouping_field or 'UNKNOWN'}\n"
            f"   - dataset cohort labels (do not confuse with metadata values): {available_groups}\n"
        )
    restrict_hint = f"6) Apply population restriction exactly: {restrict_to}\n" if restrict_to else ""
    correction_agenda = (
        "Regenerate ONLY `segmentation_request.json`.\n"
        f"Validation error: {error_message}\n\n"
        "Required:\n"
        "1) Keys present: identifiers, structures, results_database\n"
        "2) `identifiers` non-empty; format `dataset:patient:observation`\n"
        "3) Only planned observations are allowed\n"
        "4) No synthetic/placeholder identifiers\n"
        + metadata_hint
        + restrict_hint
    )
    summary_sources = build_correction_summary_sources(
        prior_paths=[phase1_json],
        current_phase_dir=save_dir,
    )
    summaries = tuple()
    if summary_sources:
        summaries = load_summaries(summary_sources)

    archive_meeting_artifacts(
        save_dir=save_dir,
        save_name="discussion",
        archive_name=f"discussion_before_correction_{correction_index:02d}",
    )

    run_meeting(
        meeting_type="individual",
        team_member=imaging_specialist,
        critic=critic,
        agenda=correction_agenda,
        summaries=summaries,
        save_dir=save_dir,
        num_rounds=1,
        temperature=temperature,
        top_p=top_p,
        prompt_verbosity=prompt_verbosity,
        enable_critic=enable_critic,
        workflow_instruction="Phase 2A: Segmentation Request Correction",
    )


def write_execution_summary(
    save_dir: Path,
    execution_summary: dict,
    request: dict,
) -> None:
    """Write execution summary to summary.md."""
    lines = ["### Execution Summary", ""]
    lines.append(f"- Success: {execution_summary.get('success', False)}")
    lines.append(f"- Total identifiers: {execution_summary.get('total_identifiers', 0)}")
    lines.append(f"- Structures: {', '.join(execution_summary.get('structures', []))}")
    lines.append(f"- Results database: {execution_summary.get('results_database', '')}")
    lines.append(f"- Cached count: {execution_summary.get('cached_count', 0)}")
    lines.append(f"- Processed count: {execution_summary.get('processed_count', 0)}")
    lines.append(f"- Model variant: {request.get('model_variant', '')}")
    lines.append(f"- Chunk size: {request.get('chunk_size', '')}")

    summary_path = save_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")


def validate_request_observations(request: dict, observations: list[str]) -> None:
    """Ensure segmentation identifiers only include planned observations."""
    allowed = set(observations)
    invalid = []
    malformed = []
    for identifier in request.get("identifiers", []):
        parts = str(identifier).split(":")
        if len(parts) != 3:
            malformed.append(identifier)
            continue
        obs = parts[2]
        if obs not in allowed:
            invalid.append(identifier)

    if malformed:
        raise ValueError(
            "Segmentation request contains malformed identifiers (expected dataset:patient:observation): "
            + ", ".join(map(str, malformed[:5]))
        )
    if invalid:
        raise ValueError(
            "Segmentation request includes observations not in plan: "
            + ", ".join(map(str, invalid[:5]))
        )


if __name__ == "__main__":
    main()
