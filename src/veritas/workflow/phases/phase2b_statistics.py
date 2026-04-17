#!/usr/bin/env python3
"""Phase 2B: Statistical Analysis

Statistician writes code to:
1. Load segmentation results from Phase 2A
2. Calculate metrics from masks
3. Perform statistical hypothesis testing
4. Save results and visualizations
"""

import os
import sys
import json
import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EXECUTION_MODE", "langgraph")
os.environ.setdefault("OLLAMA_CONTEXT_LENGTH", os.environ.get("VERITAS_CONTEXT_LENGTH", "16384"))
os.environ.setdefault("CODE_EXEC_TIMEOUT", "180")

repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from veritas import run_meeting
from veritas.prompts import (
    PHASE_AWARE_CRITIC,
    CODING_ML_STATISTICIAN_CODE_OUTPUT,
    create_agent_with_model,
)
from veritas.utils import archive_meeting_artifacts, build_correction_summary_sources, load_summaries
from veritas.verbosity import get_prompt_verbosity_config
from veritas.prompt_templates import build_phase2b_agenda
from veritas.workflow.audit import (
    append_correction_report,
    classify_failure_code,
    detect_off_contract_data_loading,
    update_workflow_audit,
    write_validation_report,
)


@dataclass
class PhaseConfig:
    """Configuration for phase execution."""
    output_path: str = "outputs/workflow"
    statistician_coding_model: str = "qwen3-coder:30b"
    critic_model: str = "qwen3:8b"
    temperature: float = 0.2
    top_p: Optional[float] = None
    # Per-role temperature overrides (None = use global temperature)
    statistician_coding_temperature: Optional[float] = None
    critic_temperature: Optional[float] = None
    # Per-role top_p overrides (None = use global top_p)
    statistician_coding_top_p: Optional[float] = None
    critic_top_p: Optional[float] = None
    prompt_verbosity: str = "standard"
    dataset_path: str = ""
    dataset_metadata: Optional[dict] = None
    sat_cache_database: Optional[str] = None
    enable_critic: bool = True


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
            statistician_coding_model=models.get("statistician_coding", data.get("statistician_coding_model", PhaseConfig.statistician_coding_model)),
            critic_model=models.get("critic", data.get("critic_model", PhaseConfig.critic_model)),
            temperature=data.get("temperature", PhaseConfig.temperature),
            top_p=data.get("top_p"),
            statistician_coding_temperature=temperatures.get("statistician_coding"),
            critic_temperature=temperatures.get("critic"),
            statistician_coding_top_p=top_ps.get("statistician_coding"),
            critic_top_p=top_ps.get("critic"),
            prompt_verbosity=data.get("prompt_verbosity", PhaseConfig.prompt_verbosity),
            dataset_path=data.get("dataset_path", ""),
            dataset_metadata=data.get("dataset"),
            sat_cache_database=data.get("sat_cache_database"),
            enable_critic=data.get("enable_critic", True),
        )

    # Fall back to environment variables
    metadata_str = os.environ.get("DATASET_METADATA")
    dataset_metadata = json.loads(metadata_str) if metadata_str else None

    return PhaseConfig(
        output_path=os.environ.get("WORKFLOW_OUTPUT_PATH", PhaseConfig.output_path),
        statistician_coding_model=os.environ.get("STATISTICIAN_CODING_MODEL", PhaseConfig.statistician_coding_model),
        critic_model=os.environ.get("CRITIC_MODEL", PhaseConfig.critic_model),
        temperature=float(os.environ.get("WORKFLOW_TEMPERATURE", str(PhaseConfig.temperature))),
        prompt_verbosity=os.environ.get("WORKFLOW_PROMPT_VERBOSITY", PhaseConfig.prompt_verbosity),
        dataset_path=os.environ.get("DATASET_PATH", ""),
        dataset_metadata=dataset_metadata,
    )


def main():
    """Run Phase 2B: Statistical analysis."""
    print("\n" + "=" * 70)
    print("PHASE 2B: STATISTICAL ANALYSIS")
    print("=" * 70 + "\n")

    config = load_config()
    config_file = os.environ.get("WORKFLOW_CONFIG_FILE")
    if not config_file or not Path(config_file).exists():
        raise RuntimeError("WORKFLOW_CONFIG_FILE not found")
    config_path_obj = Path(config_file)
    update_workflow_audit(
        config_file=config_path_obj,
        phase="phase2b",
        status="running",
        details={"script": "phase2b_statistics.py"},
    )

    save_dir: Optional[Path] = None
    correction_attempted = False
    correction_success = None
    phase2b_start_ts = None
    try:
        # Resolve output path
        output_path = Path(config.output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

        save_dir = (output_path / "phase2b_statistical_analysis").absolute()
        save_dir.mkdir(parents=True, exist_ok=True)

        workspace_dir = save_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "data").mkdir(exist_ok=True)
        (workspace_dir / "plots").mkdir(exist_ok=True)
        os.environ["CODE_EXEC_WORKSPACE_DIR"] = str(workspace_dir)

        # Set DATASET_PATH for dataset loader functions in code execution
        if config.dataset_path:
            dataset_path = Path(config.dataset_path)
            if not dataset_path.is_absolute():
                dataset_path = repo_root / dataset_path
            os.environ["DATASET_PATH"] = str(dataset_path)

        # Load summaries from previous phases
        phase1_json = (output_path / "phase1_hypothesis_formulation" / "discussion.json").absolute()
        phase2a_json = (output_path / "phase2a_imaging_analysis" / "discussion.json").absolute()
        summaries = load_summaries([p for p in [phase1_json, phase2a_json] if p.exists()])

        # Phase 2A outputs
        segmentation_execution_path = output_path / "phase2a_imaging_analysis" / "segmentation_execution.json"

        # Determine results database path.
        # Priority 1: what Phase 2A actually used (handles SAT, GT, any future source automatically)
        # Priority 2: sat_cache_database from config (fallback when Phase 2A was skipped)
        seg_exec_db = None
        if segmentation_execution_path.exists():
            with open(segmentation_execution_path) as f:
                exec_data = json.load(f)
            seg_exec_db = exec_data.get("results_database")

        if seg_exec_db:
            phase2a_results_db = Path(seg_exec_db).absolute()
        elif config.sat_cache_database:
            phase2a_results_db = Path(config.sat_cache_database)
            if not phase2a_results_db.is_absolute():
                phase2a_results_db = repo_root / phase2a_results_db
            phase2a_results_db = phase2a_results_db.absolute()
        else:
            phase2a_results_db = (output_path / "phase2a_imaging_analysis" / "results_database").absolute()

        # Expose results_db to execution engine so agent code gets it pre-injected
        os.environ["PHASE2B_RESULTS_DB"] = str(phase2a_results_db)

        # Load plan
        with open(config_file) as f:
            full_config = json.load(f)
        plan = full_config.get("plan", {})
        hypothesis_text = full_config.get("hypothesis", "")

        if not plan:
            raise ValueError("No plan found in config - Phase 1 must run first")

        # Extract plan values
        groups = plan.get("groups", [])
        structures = plan.get("structures", [])
        observations = plan.get("observations", [])
        metrics = plan.get("metrics", [])
        statistical_test = plan.get("statistical_test", "")
        cohort_mode = str(plan.get("cohort_mode", "")).strip().lower()
        restrict_to = plan.get("restrict_to") or {}
        analysis_type = plan.get("analysis_type") or infer_analysis_type(
            plan=plan,
            hypothesis=hypothesis_text,
        )
        metadata_fields = (
            config.dataset_metadata.get("patient_metadata_fields", [])
            if config.dataset_metadata else []
        )
        all_groups_selected = False
        group_tokens = [str(g).strip().lower() for g in groups]
        available_groups = (
            config.dataset_metadata.get("available_groups", [])
            if config.dataset_metadata else []
        )
        if not groups or group_tokens == ["all"] or cohort_mode == "all":
            all_groups_selected = True
            groups = available_groups or []
        predictors = normalize_predictors(plan.get("predictors"), metadata_fields)
        target_variables = normalize_target_variables(
            target_variables=plan.get("target_variables"),
            metadata_fields=metadata_fields,
        )
        if analysis_type in {"correlation", "regression"}:
            target_outcome = str(target_variables.get("outcome") or "").strip()
            target_predictors = target_variables.get("predictors") or []
            if isinstance(target_predictors, str):
                target_predictors = [target_predictors]
            target_predictors = [str(item).strip() for item in target_predictors if str(item).strip()]
            if not target_outcome or not target_predictors:
                raise ValueError(
                    "Plan missing target_variables outcome/predictors for correlation/regression analysis"
                )
            hypothesis_fields = _extract_hypothesis_metadata_fields(hypothesis_text, metadata_fields)
            if hypothesis_fields:
                target_variables["hypothesis_metadata_fields"] = hypothesis_fields
        adjust_for = normalize_confounders(plan.get("adjust_for"), metadata_fields)
        stratify_by = normalize_confounders(plan.get("stratify_by"), metadata_fields)
        adjust_for_level = resolve_adjustment_requirement_level(
            plan=plan,
            hypothesis_text=hypothesis_text,
            analysis_type=analysis_type,
        )
        stratify_by_level = resolve_stratification_requirement_level(
            plan=plan,
            hypothesis_text=hypothesis_text,
            analysis_type=analysis_type,
        )

        if not groups and analysis_type == "group_difference":
            raise ValueError("Plan missing 'groups' field")
        if analysis_type == "group_difference" and len(groups) < 2:
            raise ValueError("Group-difference analysis requires at least two groups")
        metadata_only = not structures
        if not metrics and not metadata_only:
            raise ValueError("Plan missing 'metrics' field")
        if analysis_type in {"correlation", "regression"} and not predictors:
            raise ValueError("Plan missing 'predictors' for correlation/regression analysis")

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

        # Extract domain notes
        domain_notes = config.dataset_metadata.get("domain_notes", "") if config.dataset_metadata else ""

        # Create agents
        statistician = create_agent_with_model(
            CODING_ML_STATISTICIAN_CODE_OUTPUT,
            config.statistician_coding_model,
            config.statistician_coding_temperature,
            config.statistician_coding_top_p,
        )
        critic = create_agent_with_model(
            PHASE_AWARE_CRITIC,
            config.critic_model,
            config.critic_temperature,
            config.critic_top_p,
        )

        # Build group filter instruction
        available_groups_set = set(available_groups)
        metadata_based_groups = [g for g in groups if g not in available_groups_set]
        grouping_field = plan.get("grouping_field")
        group_spec = plan.get("group_spec") if isinstance(plan.get("group_spec"), dict) else None
        if not grouping_field and group_spec and str(group_spec.get("type", "")).lower() == "metadata":
            grouping_field = group_spec.get("field")

        if all_groups_selected:
            group_filter_instruction = (
                "Do NOT filter out any patients by group; include all patients and "
                "use metadata labels to assign group membership for summaries."
            )
        elif metadata_based_groups:
            # Prefer explicit plan grouping field; otherwise detect heuristically
            metadata_group_field = None
            if isinstance(grouping_field, str) and grouping_field in metadata_fields:
                metadata_group_field = grouping_field
            else:
                metadata_group_field = _detect_metadata_group_field(
                    metadata_based_groups, metadata_fields, domain_notes
                )
            if metadata_group_field:
                group_filter_instruction = (
                    f"Groups {groups} are metadata-based. "
                    f"Get ALL patients with sat.list_patients(results_db_path), then for each "
                    f"patient get metadata with sat.get_patient_metadata(patient_id) and filter "
                    f"by metadata['{metadata_group_field}'] matching {groups}."
                )
            else:
                group_filter_instruction = (
                    f"Groups {groups} are metadata-based (not direct API groups). "
                    f"Get ALL patients with sat.list_patients(results_db_path), then for each "
                    f"patient get metadata with sat.get_patient_metadata(patient_id) and find "
                    f"the metadata field containing values {groups}. "
                    f"Available metadata fields: {metadata_fields}"
                )
        elif groups:
            group_filter_instruction = (
                f"Filter patients by groups {groups}: for each patient, get metadata['group'] from "
                f"sat.get_patient_metadata(patient_id) and match against {groups}"
            )
        else:
            group_filter_instruction = "Do NOT filter by group; include all patients with available metadata"

        agenda = build_phase2b_agenda(
            plan=plan,
            phase2a_results_db=str(phase2a_results_db),
            groups=groups,
            structures=structures,
            observations=observations,
            metrics=metrics,
            statistical_test=statistical_test,
            analysis_type=analysis_type,
            cohort_mode=cohort_mode or "groups",
            predictors=predictors,
            adjust_for=adjust_for,
            stratify_by=stratify_by,
            target_variables=target_variables,
            adjust_for_level=adjust_for_level,
            stratify_by_level=stratify_by_level,
            metadata_fields=metadata_fields,
            domain_notes=domain_notes,
            group_filter_instruction=group_filter_instruction,
            all_groups_selected=all_groups_selected,
            restrict_to=restrict_to,
        )

        phase2b_run_uuid = str(uuid.uuid4())
        phase2b_start_ts = time.time()
        provenance_manifest_path = workspace_dir / "data" / "phase2b_run_manifest.json"
        provenance_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_manifest = {
            "phase": "phase2b_statistics",
            "run_uuid": phase2b_run_uuid,
            "phase2b_start_ts": phase2b_start_ts,
            "expected_results_path": str((workspace_dir / "data" / "statistical_results.json").resolve()),
        }
        provenance_manifest_path.write_text(json.dumps(provenance_manifest, indent=2))

        print(f"Running statistical analysis with:")
        print(f"  • Statistician: {config.statistician_coding_model}")
        print(f"  • Critic: {config.critic_model}")
        print(f"  • Temperature: {config.temperature}")
        print(f"  • Verbosity: {config.prompt_verbosity}")
        print()

        checklist_items = [
            "`data/statistical_results.json` exists in `data/`.",
            "JSON includes required keys: analysis_type, test_performed, p_value, effect_size, effect_size_type, n_total, sample_sizes; and `n_total` is a positive integer.",
            "Use the full eligible cohort after restrictions (no manual capping/subsampling).",
            "Final response cites exact values from `data/statistical_results.json`.",
        ]
        if analysis_type in {"correlation", "regression"}:
            checklist_items.append(
                "`statistical_results.json` includes `variables_tested` with exact planned outcome + predictors."
            )
        if analysis_type == "survival":
            checklist_items.append(
                "For survival analyses, set `effect_size_type` to `hazard_ratio`."
            )
        checklist_lines = "\n".join(
            f"{index}) {item}" for index, item in enumerate(checklist_items, start=1)
        )
        checklist_prompt = (
            "**Final output checklist (must be satisfied before your final response):**\n"
            + checklist_lines
            + "\nIf any item fails, continue debugging instead of finalizing."
        )

        run_meeting(
            meeting_type="individual",
            team_member=statistician,
            critic=critic,
            agenda=agenda,
            summaries=summaries,
            save_dir=save_dir,
            num_rounds=1,
            temperature=config.temperature,
            top_p=config.top_p,
            prompt_verbosity=config.prompt_verbosity,
            enable_critic=config.enable_critic,
            workflow_instruction="Phase 2B: Statistical Analysis - Analyze results and test hypothesis",
            summary_instructions=checklist_prompt,
        )

        print(f"\n✓ Phase 2B complete. Results saved to {save_dir}")
        print(f"  • Discussion: {save_dir / 'discussion.json'}")
        print(f"  • Summary: {save_dir / 'summary.md'}")

        # Validate outputs with up to two correction passes on failure
        validation_error_first: Optional[str] = None
        validated = False
        max_corrections = 2
        for correction_index in range(max_corrections + 1):
            try:
                validate_analysis_outputs(
                    workspace_dir,
                    required_adjust_for=adjust_for,
                    required_stratify_by=stratify_by,
                    adjustment_requirement_level=adjust_for_level,
                    stratification_requirement_level=stratify_by_level,
                    metadata_fields=metadata_fields,
                    expected_analysis_type=analysis_type,
                    planned_target_variables=target_variables,
                    hypothesis_metadata_fields=(
                        target_variables.get("hypothesis_metadata_fields")
                        if isinstance(target_variables, dict)
                        else None
                    ),
                    provenance_manifest_path=provenance_manifest_path,
                )
                validated = True
                break
            except (FileNotFoundError, ValueError) as error:
                if validation_error_first is None:
                    validation_error_first = str(error)
                if correction_index >= max_corrections:
                    correction_success = False if correction_attempted else None
                    if correction_attempted:
                        append_correction_report(
                            save_dir=save_dir,
                            phase="phase2b",
                            attempted=True,
                            success=False,
                            reason="statistical_results_still_invalid_after_corrections",
                            error_before=validation_error_first,
                            error_after=str(error),
                        )
                    raise

                correction_attempted = True
                print(f"\n  ⚠ Statistical output validation failed: {error}")
                print(
                    f"  → Requesting correction pass {correction_index + 1}/{max_corrections} "
                    "from Statistician..."
                )
                _request_statistics_output_correction(
                    save_dir=save_dir,
                    workspace_dir=workspace_dir,
                    phase1_json=phase1_json,
                    phase2a_json=phase2a_json,
                    statistician=statistician,
                    critic=critic,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    prompt_verbosity=config.prompt_verbosity,
                    enable_critic=config.enable_critic,
                    error_message=str(error),
                    correction_index=correction_index + 1,
                )

        if correction_attempted and validated:
            correction_success = True
            append_correction_report(
                save_dir=save_dir,
                phase="phase2b",
                attempted=True,
                success=True,
                reason="statistical_results_correction_succeeded",
                error_before=validation_error_first,
            )

        print(f"  • Statistical results: {workspace_dir / 'data' / 'statistical_results.json'}")
        print(f"  • Plots: {workspace_dir / 'plots'}")

        write_validation_report(
            save_dir=save_dir,
            phase="phase2b",
            status="passed",
            checks={
                "statistical_results_present": True,
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase2b",
            status="passed",
            details={
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
    except Exception as e:
        failure_code = classify_failure_code("phase2b", str(e))
        if save_dir:
            write_validation_report(
                save_dir=save_dir,
                phase="phase2b",
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
            phase="phase2b",
            status="failed",
            failure_code=failure_code,
            failure_message=str(e),
            details={
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        raise


def validate_analysis_outputs(
    workspace_dir: Path,
    required_adjust_for: Optional[list[str]] = None,
    required_stratify_by: Optional[list[str]] = None,
    adjustment_requirement_level: str = "none",
    stratification_requirement_level: str = "none",
    metadata_fields: Optional[list[str]] = None,
    expected_analysis_type: Optional[str] = None,
    planned_target_variables: Optional[dict] = None,
    hypothesis_metadata_fields: Optional[list[str]] = None,
    provenance_manifest_path: Optional[Path] = None,
) -> None:
    """Validate that required analysis outputs exist."""
    results_file = workspace_dir / "data" / "statistical_results.json"
    mtime_tolerance_seconds = 3.0

    if not results_file.exists():
        raise FileNotFoundError(
            f"Statistical results not found: {results_file}\n"
            "Agent must write statistical_results.json to data/"
        )

    if provenance_manifest_path and provenance_manifest_path.exists():
        with open(provenance_manifest_path) as manifest_file:
            manifest = json.load(manifest_file)
        expected_path = manifest.get("expected_results_path")
        if expected_path and str(results_file.resolve()) != str(Path(expected_path).resolve()):
            raise ValueError(
                "Statistical results path does not match provenance manifest expected path: "
                f"{expected_path}"
            )
        manifest_start_ts = manifest.get("phase2b_start_ts")
        if manifest_start_ts is not None:
            try:
                manifest_start_ts_float = float(manifest_start_ts)
            except Exception:
                raise ValueError("Invalid phase2b provenance manifest timestamps")
            if results_file.stat().st_mtime + mtime_tolerance_seconds < manifest_start_ts_float:
                raise ValueError(
                    "Statistical results file appears stale (mtime older than phase2b_start_ts)"
                )

    with open(results_file) as f:
        results = json.load(f)

    required = ["analysis_type", "test_performed", "p_value", "effect_size", "effect_size_type", "n_total", "sample_sizes"]
    missing = [key for key in required if key not in results]
    if missing:
        raise ValueError(
            f"Statistical results missing required keys: {', '.join(missing)}"
        )

    analysis_type = results.get("analysis_type")
    analysis_type_norm = str(analysis_type or "").strip().lower()
    expected_analysis_type_norm = str(expected_analysis_type or "").strip().lower()
    metadata_fields = metadata_fields or []
    if expected_analysis_type_norm and analysis_type_norm and analysis_type_norm != expected_analysis_type_norm:
        print(
            "  ⚠ Warning: analysis_type in statistical_results.json does not match Phase 1 plan "
            f"({analysis_type_norm} vs {expected_analysis_type_norm})"
        )
    if analysis_type == "group_difference" and "group_statistics" not in results:
        print(
            "  ⚠ Warning: group_statistics missing for group_difference analysis "
            "— Phase 3 should note this limitation"
        )

    n_total = results.get("n_total")
    try:
        if n_total is None or int(n_total) <= 0:
            raise ValueError
    except Exception:
        raise ValueError("n_total must be a positive integer")

    effect_size_type = str(results.get("effect_size_type", "")).lower().replace("-", "_")
    test_name = str(results.get("test_performed", "")).lower()
    if "mann" in test_name and "whitney" in test_name and effect_size_type not in {"rank_biserial", "rank_biserial_correlation"}:
        print("  ⚠ Warning: effect_size_type should be 'rank_biserial' for Mann-Whitney")
    if ("t-test" in test_name or "ttest" in test_name) and effect_size_type not in {"cohens_d", "cohen_d", "cohen's_d"}:
        print("  ⚠ Warning: effect_size_type should be 'cohens_d' for t-test")
    if effect_size_type in {"rank_biserial", "rank_biserial_correlation"}:
        try:
            if abs(float(results.get("effect_size"))) > 1.0:
                print("  ⚠ Warning: rank-biserial effect size should be within [-1, 1]")
        except (TypeError, ValueError):
            print("  ⚠ Warning: effect_size is not a valid number")

    if "confidence_interval" in results and "confidence_interval_type" not in results:
        print("  ⚠ Warning: confidence_interval_type missing (effect_size vs mean_difference)")
    if "confidence_interval_type" in results:
        ci_type = str(results.get("confidence_interval_type")).strip().lower()
        if ci_type not in {"effect_size", "mean_difference"}:
            print("  ⚠ Warning: confidence_interval_type should be 'effect_size' or 'mean_difference'")
        if ci_type == "mean_difference" and "confidence_interval_groups" not in results:
            print("  ⚠ Warning: confidence_interval_groups missing for mean_difference CI")

    if analysis_type_norm == "survival":
        if any(token in test_name for token in ("mann", "whitney", "t-test", "ttest")):
            raise ValueError(
                "Survival analysis cannot use t-test/Mann-Whitney. Use log-rank and/or Cox PH."
            )
        try:
            p_value = float(results.get("p_value"))
        except Exception:
            raise ValueError("Survival analysis requires a numeric p_value")
        if not (0.0 <= p_value <= 1.0):
            raise ValueError("Survival analysis p_value must be within [0, 1]")
        try:
            hr = float(results.get("effect_size"))
        except Exception:
            raise ValueError("Survival analysis requires numeric effect_size (hazard ratio)")
        if hr <= 0:
            raise ValueError("Survival analysis hazard ratio must be > 0")
        if effect_size_type != "hazard_ratio":
            raise ValueError(
                "Survival analysis requires effect_size_type='hazard_ratio' "
                "(do not store test statistics as effect_size)"
            )
        ci = results.get("confidence_interval")
        if isinstance(ci, dict):
            lower = ci.get("lower")
            upper = ci.get("upper")
            if lower is not None and upper is not None:
                try:
                    lower_f = float(lower)
                    upper_f = float(upper)
                    if lower_f >= upper_f:
                        raise ValueError("Survival confidence_interval must satisfy lower < upper")
                    if abs(hr - 1.0) < 1e-8 and abs(lower_f - 1.0) < 1e-8 and abs(upper_f - 1.0) < 1e-8:
                        raise ValueError("Degenerate survival result detected (HR=1.0 with CI=[1.0,1.0])")
                except ValueError:
                    raise
                except Exception:
                    raise ValueError("Survival confidence_interval bounds must be numeric when provided")

    enforce_target_contract = (
        analysis_type_norm in {"correlation", "regression"}
        or expected_analysis_type_norm in {"correlation", "regression"}
    )
    if enforce_target_contract:
        variables_tested = results.get("variables_tested")
        if not isinstance(variables_tested, dict):
            raise ValueError(
                "Correlation/regression results require `variables_tested` object "
                "with keys: outcome, predictors."
            )

        output_outcome = str(variables_tested.get("outcome") or "").strip()
        output_predictors = variables_tested.get("predictors")
        if isinstance(output_predictors, str):
            output_predictors = [output_predictors]
        if not isinstance(output_predictors, list):
            output_predictors = []
        output_predictors = [str(item).strip() for item in output_predictors if str(item).strip()]

        if not output_outcome:
            raise ValueError("variables_tested.outcome must be a non-empty string.")
        if len(output_predictors) == 0:
            raise ValueError("variables_tested.predictors must contain at least one variable.")

        planned = planned_target_variables if isinstance(planned_target_variables, dict) else {}
        planned_outcome = str(planned.get("outcome") or "").strip()
        planned_predictors = planned.get("predictors") or []
        if isinstance(planned_predictors, str):
            planned_predictors = [planned_predictors]
        planned_predictors = [
            str(item).strip()
            for item in planned_predictors
            if str(item).strip()
        ]

        def canonicalize(value: str) -> str:
            resolved = _resolve_metadata_field(value, metadata_fields)
            return resolved or str(value).strip()

        if planned_outcome and canonicalize(output_outcome).lower() != canonicalize(planned_outcome).lower():
            raise ValueError(
                "variables_tested.outcome must match planned target_variables.outcome "
                f"(expected '{planned_outcome}', got '{output_outcome}')."
            )

        if planned_predictors:
            planned_norm = {canonicalize(value).lower() for value in planned_predictors}
            output_norm = {canonicalize(value).lower() for value in output_predictors}
            if planned_norm != output_norm:
                raise ValueError(
                    "variables_tested.predictors must match planned target_variables.predictors "
                    f"(expected {planned_predictors}, got {output_predictors})."
                )

        hypothesis_fields = []
        if isinstance(hypothesis_metadata_fields, list):
            hypothesis_fields = [str(field).strip() for field in hypothesis_metadata_fields if str(field).strip()]

        if hypothesis_fields:
            variable_pool = {canonicalize(output_outcome)}
            variable_pool.update(canonicalize(value) for value in output_predictors)
            missing_hypothesis_fields = [
                field for field in hypothesis_fields
                if canonicalize(field) not in variable_pool
            ]
            if missing_hypothesis_fields:
                raise ValueError(
                    "variables_tested does not include metadata variables named in the hypothesis: "
                    + ", ".join(missing_hypothesis_fields)
                )

    if required_adjust_for or required_stratify_by:
        adjustment_level_norm = normalize_requirement_level(adjustment_requirement_level)
        stratification_level_norm = normalize_requirement_level(stratification_requirement_level)
        available_fields = metadata_fields or []
        required_adjust_for = normalize_confounders(required_adjust_for, available_fields)
        required_stratify_by = normalize_confounders(required_stratify_by, available_fields)
        reported_adjust_for = normalize_confounders(results.get("adjusted_for"), available_fields)
        reported_stratify_by = normalize_confounders(results.get("stratified_by"), available_fields)

        if required_adjust_for and adjustment_level_norm in {"required", "recommended"}:
            missing_adjust = [field for field in required_adjust_for if field not in reported_adjust_for]
            if missing_adjust:
                if adjustment_level_norm == "required":
                    raise ValueError(
                        "Adjusted analysis contract violated: statistical_results.json missing adjust_for fields: "
                        + ", ".join(missing_adjust)
                    )
                print(
                    "  ⚠ Warning: statistical_results.json missing recommended adjust_for fields: "
                    + ", ".join(missing_adjust)
                    + " — Phase 3 should note this limitation"
                )

        if required_stratify_by and stratification_level_norm in {"required", "recommended"}:
            missing_strata = [field for field in required_stratify_by if field not in reported_stratify_by]
            if missing_strata:
                if stratification_level_norm == "required":
                    raise ValueError(
                        "Stratification contract violated: statistical_results.json missing stratify_by fields: "
                        + ", ".join(missing_strata)
                    )
                print(
                    "  ⚠ Warning: statistical_results.json missing recommended stratify_by fields: "
                    + ", ".join(missing_strata)
                    + " — Phase 3 should note this limitation"
                )

    if required_adjust_for and normalize_requirement_level(adjustment_requirement_level) in {"required", "recommended"} and analysis_type in {"correlation", "regression"}:
        effect_size_type = str(results.get("effect_size_type", "")).lower().replace("-", "_")
        adjusted_effect_types = {
            "partial_pearson_r",
            "partial_spearman_r",
            "regression_beta",
            "regression_beta_std",
            "regression_coef",
        }
        raw_effect_types = {"pearson_r", "spearman_rho"}
        adjusted_block = results.get("adjusted_effect_size")

        has_adjusted_block = False
        if isinstance(adjusted_block, dict):
            block_type = str(adjusted_block.get("effect_size_type", "")).lower().replace("-", "_")
            if block_type and block_type not in raw_effect_types:
                has_adjusted_block = True

        if effect_size_type in raw_effect_types and not has_adjusted_block:
            print(
                "  ⚠ Warning: adjust_for is set but effect_size_type is unadjusted. "
                "Expected partial_* or regression_* effect_size_type"
            )
        if effect_size_type and effect_size_type not in raw_effect_types and effect_size_type not in adjusted_effect_types:
            if not has_adjusted_block:
                print(
                    "  ⚠ Warning: effect_size_type is not recognized for adjusted analysis. "
                    "Expected partial_* or regression_* effect_size_type"
                )

    if required_stratify_by and normalize_requirement_level(stratification_requirement_level) in {"required", "recommended"} and analysis_type in {"correlation", "regression"}:
        stratified_results = results.get("stratified_results")
        if not isinstance(stratified_results, dict) or not stratified_results:
            print(
                "  ⚠ Warning: statistical_results.json missing stratified_results "
                "— Phase 3 should note this limitation"
            )
        else:
            sample_sizes = results.get("sample_sizes") or {}
            if isinstance(sample_sizes, dict) and sample_sizes:
                missing_groups = [group for group in sample_sizes if group not in stratified_results]
                if missing_groups:
                    print(
                        "  ⚠ Warning: stratified_results missing entries for groups: "
                        + ", ".join(missing_groups)
                    )

    plots_dir = workspace_dir / "plots"
    if not plots_dir.exists() or not list(plots_dir.glob("*.png")):
        print("  ⚠ Warning: No plots found in workspace/plots/")

    validation_code_files = _get_active_validation_code_files(workspace_dir)

    synthetic_scan = detect_synthetic_data_usage(
        workspace_dir,
        code_files=validation_code_files,
    )
    hard_hits = synthetic_scan.get("hard_hits", [])
    soft_hits = synthetic_scan.get("soft_hits", [])
    if soft_hits:
        formatted_soft = "; ".join(
            f"{hit['file']}:{hit['line']} [{hit['rule']}]"
            for hit in soft_hits[:3]
        )
        print(
            "  ⚠ Warning: Non-blocking random usage detected in Phase 2B code "
            f"(allowed if used for plotting jitter/bootstrap only). Examples: {formatted_soft}"
        )
    if hard_hits:
        formatted = "; ".join(
            f"{hit['file']}:{hit['line']} [{hit['rule']}]"
            for hit in hard_hits[:5]
        )
        raise ValueError(
            "Synthetic/mock data usage detected in Phase 2B analysis code "
            f"(violates prompt policy). Examples: {formatted}"
        )

    off_contract_scan = detect_off_contract_data_loading(
        workspace_dir / "code",
        code_files=validation_code_files,
    )
    off_contract_soft_hits = off_contract_scan.get("soft_hits", [])
    off_contract_hard_hits = off_contract_scan.get("hard_hits", [])
    if off_contract_soft_hits:
        formatted_soft = "; ".join(
            f"{hit['file']}:{hit['line']} [{hit['rule']}]"
            for hit in off_contract_soft_hits[:3]
        )
        print(
            "  ⚠ Warning: Possible off-contract data loading patterns found in files that also use SAT "
            f"(review recommended). Examples: {formatted_soft}"
        )
    if off_contract_hard_hits:
        formatted_hard = "; ".join(
            f"{hit['file']}:{hit['line']} [{hit['rule']}]"
            for hit in off_contract_hard_hits[:5]
        )
        raise ValueError(
            "Off-contract data loading detected in Phase 2B analysis code "
            f"(primary source must be SAT APIs). Examples: {formatted_hard}"
        )

    sample_cap_hits = detect_sample_capping_usage(
        workspace_dir,
        code_files=validation_code_files,
    )
    if sample_cap_hits:
        hard_cap_hits = [hit for hit in sample_cap_hits if hit.get("severity") == "hard"]
        soft_cap_hits = [hit for hit in sample_cap_hits if hit.get("severity") != "hard"]
        if soft_cap_hits:
            formatted_soft_caps = "; ".join(
                f"{hit['file']}:{hit['line']} [{hit['rule']}]"
                for hit in soft_cap_hits[:5]
            )
            print(
                "  ⚠ Warning: Possible sample capping/subsampling pattern detected in Phase 2B code. "
                f"Examples: {formatted_soft_caps}"
            )
        if hard_cap_hits:
            formatted_hard_caps = "; ".join(
                f"{hit['file']}:{hit['line']} [{hit['rule']}]"
                for hit in hard_cap_hits[:5]
            )
            raise ValueError(
                "Sample capping/subsampling detected in Phase 2B analysis code "
                f"(full cohort is required). Examples: {formatted_hard_caps}"
            )


def _resolve_phase2b_code_path(workspace_dir: Path, filename: str) -> Optional[Path]:
    try:
        raw_path = Path(str(filename).strip())
        candidate = raw_path if raw_path.is_absolute() else workspace_dir / raw_path
        resolved = candidate.resolve()
        workspace_root = workspace_dir.resolve()
        resolved.relative_to(workspace_root)
    except Exception:
        return None

    if resolved.suffix != ".py" or not resolved.exists():
        return None
    return resolved


def _get_active_validation_code_files(workspace_dir: Path) -> list[Path]:
    code_dir = workspace_dir / "code"
    default_files = sorted(code_dir.rglob("*.py")) if code_dir.exists() else []
    if not default_files:
        return []

    metadata_path = workspace_dir.parent / "discussion_execution_metadata.json"
    if not metadata_path.exists():
        return default_files

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception:
        return default_files

    executions = metadata.get("executions")
    if not isinstance(executions, list) or not executions:
        return default_files

    selected_filename: Optional[str] = None
    for execution in reversed(executions):
        if isinstance(execution, dict) and execution.get("success") and execution.get("filename"):
            selected_filename = str(execution.get("filename"))
            break
    if selected_filename is None:
        for execution in reversed(executions):
            if isinstance(execution, dict) and execution.get("filename"):
                selected_filename = str(execution.get("filename"))
                break

    if not selected_filename:
        return default_files

    selected_path = _resolve_phase2b_code_path(workspace_dir, selected_filename)
    if selected_path is None:
        return default_files

    rel_path = str(selected_path.relative_to(workspace_dir))
    print(f"  • Validation scope: scanning active executed file `{rel_path}`")
    return [selected_path]


def detect_sample_capping_usage(workspace_dir: Path, code_files: Optional[list[Path]] = None) -> list[dict]:
    """Detect likely patient-level subsampling/capping patterns in analysis code."""
    code_dir = workspace_dir / "code"
    if not code_dir.exists():
        return []

    patterns = [
        ("slice_cap", re.compile(r"\[\s*:\s*(\d{1,4})\s*\]")),
        ("min_len_cap", re.compile(r"\bmin\s*\(\s*(\d{1,4})\s*,\s*len\s*\(")),
        ("head_cap", re.compile(r"\.head\s*\(\s*(\d{1,4})\s*\)")),
        ("sample_cap", re.compile(r"\.sample\s*\(\s*(?:n\s*=\s*)?(\d{1,4})")),
        ("max_patients_cap", re.compile(r"\bmax_patients\b\s*=\s*(\d{1,4})")),
        ("limit_comment", re.compile(r"\blimit(?:ed)?\s+to\s+\d{1,4}\b", re.IGNORECASE)),
    ]
    context_tokens = ("patient", "patients", "cohort", "subject", "group", "grade", "ids")
    hard_rules = {"max_patients_cap"}

    if code_files is None:
        files_to_scan = sorted(code_dir.rglob("*.py"))
    else:
        files_to_scan = [path for path in code_files if path.exists() and path.suffix == ".py"]

    hits: list[dict] = []
    for py_file in files_to_scan:
        try:
            lines = py_file.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        try:
            rel = str(py_file.relative_to(workspace_dir))
        except Exception:
            rel = str(py_file)
        for lineno, line in enumerate(lines, start=1):
            lowered = line.lower()
            window = " ".join(
                l.lower() for l in lines[max(0, lineno - 2): min(len(lines), lineno + 2)]
            )
            if not any(token in window for token in context_tokens):
                continue
            for rule_name, pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                if match.groups():
                    try:
                        cap_value = int(match.group(1))
                    except Exception:
                        cap_value = None
                    if cap_value is not None and cap_value <= 0:
                        continue
                hits.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "rule": rule_name,
                        "severity": "hard" if rule_name in hard_rules else "soft",
                        "text": line.strip()[:220],
                    }
                )
                break
    return hits


def _is_plot_jitter_random_usage(window_lower: str, line_lower: str) -> bool:
    jitter_tokens = ("jitter", "x_jitter", "swarm", "strip", "beeswarm")
    plot_tokens = ("plt.scatter", "scatter(", "stripplot", "swarmplot", "boxplot")
    return any(tok in window_lower for tok in jitter_tokens) or (
        any(tok in window_lower for tok in plot_tokens) and "np.random.normal" in line_lower
    )


def _is_resampling_random_usage(window_lower: str, line_lower: str) -> bool:
    if "np.random.choice" not in line_lower and "np.random.permutation" not in line_lower and "np.random.shuffle" not in line_lower:
        return False
    resampling_tokens = ("bootstrap", "bootstrapp", "resampl", "permutation", "permute", "shuffle")
    return any(tok in window_lower for tok in resampling_tokens)


def _extract_random_assignment_target(line_lower: str) -> Optional[str]:
    match = re.match(
        r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:np\.random\.|random\.|torch\.)",
        line_lower,
    )
    if not match:
        return None
    return match.group(1).strip().lower()


def _looks_like_synthetic_data_target(var_name: Optional[str]) -> bool:
    if not var_name:
        return False
    safe_tokens = ("jitter", "seed", "perm", "bootstrap", "noise")
    if any(token in var_name for token in safe_tokens):
        return False
    suspicious_tokens = (
        "data",
        "dataset",
        "df",
        "array",
        "values",
        "volume",
        "metric",
        "feature",
        "survival",
        "group",
        "patients",
        "cohort",
    )
    return any(token in var_name for token in suspicious_tokens)


def _classify_placeholder_usage(window_lower: str, line_lower: str) -> str:
    """Classify placeholder comments: hard only for analysis/measurement placeholders."""
    if "placeholder" not in line_lower:
        return "none"
    benign_tokens = ("p-value", "p values", "confidence interval", "ci_", "ci ", "effect size", "for each group")
    if any(tok in window_lower for tok in benign_tokens):
        return "soft"
    hard_tokens = ("calculation", "derive", "derived", "measurement", "metric", "feature", "gls", "volume", "data")
    if any(tok in window_lower for tok in hard_tokens):
        return "hard"
    return "soft"


def detect_synthetic_data_usage(workspace_dir: Path, code_files: Optional[list[Path]] = None) -> dict:
    """Detect synthetic/mock-data generation patterns in Phase 2B code artifacts.

    Returns both hard violations and soft random-usage hits (e.g., plotting jitter/bootstrap).
    """
    code_dir = workspace_dir / "code"
    if not code_dir.exists():
        return {"hard_hits": [], "soft_hits": []}

    phrase_rules = [
        ("simulate_data", re.compile(r"\bsimulat(?:e|es|ed|ing)\b")),
        ("mock_data", re.compile(r"\bmock data\b")),
        ("synthetic_data", re.compile(r"\bsynthetic data\b")),
        ("demo_data", re.compile(r"\bfor demonstration\b")),
    ]
    np_random_any = re.compile(r"\bnp\.random\.(\w+)\b")
    py_random_any = re.compile(r"(?<!\.)\brandom\.(\w+)\b")
    torch_random_any = re.compile(r"\btorch\.(rand|randn|randint)\b")
    allowed_np_random = {"seed", "choice", "permutation", "shuffle"}
    allowed_py_random = {"seed", "shuffle", "sample", "choices"}

    if code_files is None:
        files_to_scan = sorted(code_dir.rglob("*.py"))
    else:
        files_to_scan = [path for path in code_files if path.exists() and path.suffix == ".py"]

    hard_hits: list[dict] = []
    soft_hits: list[dict] = []
    for py_file in files_to_scan:
        try:
            lines = py_file.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        try:
            rel = str(py_file.relative_to(workspace_dir))
        except Exception:
            rel = str(py_file)
        for lineno, line in enumerate(lines, start=1):
            lowered = line.lower()
            window = " ".join(
                l.lower()
                for l in lines[max(0, lineno - 3): min(len(lines), lineno + 2)]
            )

            for rule_name, phrase in phrase_rules:
                if phrase.search(lowered):
                    hard_hits.append({"file": rel, "line": lineno, "rule": rule_name, "text": line.strip()[:200]})

            placeholder_class = _classify_placeholder_usage(window, lowered)
            if placeholder_class == "hard":
                hard_hits.append({"file": rel, "line": lineno, "rule": "placeholder_analysis", "text": line.strip()[:200]})
            elif placeholder_class == "soft":
                soft_hits.append({"file": rel, "line": lineno, "rule": "placeholder_nonblocking", "text": line.strip()[:200]})

            np_match = np_random_any.search(line)
            if np_match:
                fn = np_match.group(1).lower()
                target_var = _extract_random_assignment_target(lowered)
                if fn in allowed_np_random:
                    if fn == "seed":
                        soft_hits.append({"file": rel, "line": lineno, "rule": "rng_seed", "text": line.strip()[:200]})
                    elif _is_plot_jitter_random_usage(window, lowered):
                        soft_hits.append({"file": rel, "line": lineno, "rule": "rng_plot_jitter", "text": line.strip()[:200]})
                    elif _is_resampling_random_usage(window, lowered):
                        soft_hits.append({"file": rel, "line": lineno, "rule": "rng_resampling", "text": line.strip()[:200]})
                    else:
                        soft_hits.append({"file": rel, "line": lineno, "rule": "rng_nonblocking", "text": line.strip()[:200]})
                elif _is_plot_jitter_random_usage(window, lowered):
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_plot_jitter", "text": line.strip()[:200]})
                elif _is_resampling_random_usage(window, lowered):
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_resampling", "text": line.strip()[:200]})
                elif _looks_like_synthetic_data_target(target_var):
                    hard_hits.append({"file": rel, "line": lineno, "rule": "np_random_assigned_data", "text": line.strip()[:200]})
                else:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_distribution_review", "text": line.strip()[:200]})

            py_match = py_random_any.search(line)
            if py_match:
                fn = py_match.group(1).lower()
                target_var = _extract_random_assignment_target(lowered)
                if fn in allowed_py_random:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_nonblocking", "text": line.strip()[:200]})
                elif _looks_like_synthetic_data_target(target_var):
                    hard_hits.append({"file": rel, "line": lineno, "rule": "py_random_assigned_data", "text": line.strip()[:200]})
                else:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_distribution_review", "text": line.strip()[:200]})

            if torch_random_any.search(line):
                target_var = _extract_random_assignment_target(lowered)
                if _looks_like_synthetic_data_target(target_var):
                    hard_hits.append({"file": rel, "line": lineno, "rule": "torch_random_assigned_data", "text": line.strip()[:200]})
                else:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_distribution_review", "text": line.strip()[:200]})

    return {"hard_hits": hard_hits, "soft_hits": soft_hits}


def _request_statistics_output_correction(
    correction_index: int,
    *,
    save_dir: Path,
    workspace_dir: Path,
    phase1_json: Path,
    phase2a_json: Path,
    statistician,
    critic,
    temperature: float,
    top_p: Optional[float] = None,
    prompt_verbosity: str,
    enable_critic: bool,
    error_message: str,
) -> None:
    targeted_hints = _build_phase2b_correction_hints(error_message)
    targeted_hints_text = "\n".join(f"- {hint}" for hint in targeted_hints)
    current_snapshot = _load_phase2b_results_snapshot(workspace_dir)
    correction_agenda = (
        "Fix Phase 2B outputs in workspace.\n"
        f"Validation error: {error_message}\n\n"
        "Targeted fixes for this error:\n"
        f"{targeted_hints_text}\n\n"
        "Required:\n"
        "1) Write `data/statistical_results.json`\n"
        "2) Include keys: analysis_type, test_performed, p_value, effect_size, effect_size_type, n_total, sample_sizes\n"
        "3) `n_total` must be a positive integer\n"
        "4) No synthetic/mock placeholder data\n"
        "5) No patient subsampling/capping (no [:N], head(N), sample(n=N), max_patients)\n"
        "6) `results_db_path` is pre-loaded in your environment — use it directly, do NOT redefine it with hardcoded paths\n"
        "7) If correlation/regression, include `variables_tested` with exact outcome + predictors used\n"
        "8) If SAT calls fail, debug SAT usage; do not replace with simulated data\n\n"
        "Current `data/statistical_results.json` snapshot (if present):\n"
        f"{current_snapshot}\n"
    )
    summary_sources = build_correction_summary_sources(
        prior_paths=[phase1_json, phase2a_json],
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
        team_member=statistician,
        critic=critic,
        agenda=correction_agenda,
        summaries=summaries,
        save_dir=save_dir,
        num_rounds=1,
        temperature=temperature,
        top_p=top_p,
        prompt_verbosity=prompt_verbosity,
        enable_critic=enable_critic,
        workflow_instruction="Phase 2B: Statistical Output Correction",
    )


def _build_phase2b_correction_hints(error_message: str) -> list[str]:
    msg = str(error_message or "").lower()
    hints: list[str] = []

    if "statistical results not found" in msg:
        hints.append("Write exactly `data/statistical_results.json` (not `cox_model_results.json` or other filenames).")
    if "missing required keys" in msg:
        hints.append("Add every missing required key explicitly to the JSON output contract.")
    if "numeric p_value" in msg:
        hints.append("Set `p_value` to a numeric float in [0, 1] from the actual fitted test.")
    if "numeric effect_size" in msg and "hazard ratio" in msg:
        hints.append("For survival, set `effect_size` to numeric hazard ratio (HR) from a Cox model fit.")
    if "effect_size_type='hazard_ratio'" in msg or "hazard ratio" in msg:
        hints.append("For survival, set `effect_size_type` exactly to `hazard_ratio`.")
    if "n_total must be a positive integer" in msg:
        hints.append("Set `n_total` to the analyzed cohort size after filtering/cleaning; it must be > 0.")
    if "variables_tested" in msg:
        hints.append(
            "For correlation/regression, set `variables_tested` with exact outcome and predictors "
            "and keep them consistent with the Phase 1 plan."
        )
    if "adjusted analysis contract violated" in msg or "missing adjust_for fields" in msg:
        hints.append(
            "When adjustment is required, include all plan covariates in the fitted model and set "
            "`adjusted_for` in `statistical_results.json` to exactly those covariates."
        )
    if "stratification contract violated" in msg or "missing stratify_by fields" in msg:
        hints.append(
            "When stratification is required, set `stratified_by` exactly as planned and provide "
            "`stratified_results` per stratum."
        )
    if "synthetic/mock data usage" in msg or "synthetic" in msg or "mock" in msg:
        hints.append("Remove any simulated/mock/randomly generated cohorts; use SAT API-derived patient data only.")
    if "subsampling" in msg or "max_patients" in msg or "head(" in msg or "sample(" in msg:
        hints.append("Remove manual capping/subsampling; run on the full eligible cohort.")

    if not hints:
        hints.append("Fix only the reported contract violation and keep the planned analysis/test unchanged.")
    return hints


def _load_phase2b_results_snapshot(workspace_dir: Path) -> str:
    results_file = workspace_dir / "data" / "statistical_results.json"
    if not results_file.exists():
        return "(missing)"
    try:
        with open(results_file) as f:
            data = json.load(f)
        snapshot_keys = [
            "analysis_type",
            "test_performed",
            "p_value",
            "effect_size",
            "effect_size_type",
            "n_total",
            "sample_sizes",
            "adjusted_for",
            "stratified_by",
            "variables_tested",
        ]
        snapshot = {key: data.get(key) for key in snapshot_keys if key in data}
        return json.dumps(snapshot, indent=2)
    except Exception as exc:
        return f"(unreadable: {exc})"


def normalize_requirement_level(level: Optional[str]) -> str:
    value = str(level or "none").strip().lower()
    if value in {"required", "recommended", "none"}:
        return value
    return "none"


def _has_adjustment_wording(hypothesis_text: str) -> bool:
    text = str(hypothesis_text or "").lower()
    return any(
        token in text
        for token in (
            "adjusting for",
            "adjusted for",
            "when accounting for",
            "accounting for",
            "controlling for",
            "after adjusting",
        )
    )


def resolve_adjustment_requirement_level(
    plan: dict,
    hypothesis_text: str,
    analysis_type: str,
) -> str:
    adjust_for = plan.get("adjust_for") or []
    if isinstance(adjust_for, str):
        adjust_for = [adjust_for]
    if not adjust_for:
        return "none"
    if _has_adjustment_wording(hypothesis_text):
        return "required"
    if analysis_type == "survival":
        return "recommended"
    if analysis_type in {"correlation", "regression"}:
        return "recommended"
    return "none"


def resolve_stratification_requirement_level(
    plan: dict,
    hypothesis_text: str,
    analysis_type: str,
) -> str:
    stratify_by = plan.get("stratify_by") or []
    if isinstance(stratify_by, str):
        stratify_by = [stratify_by]
    if not stratify_by:
        return "none"
    if "stratif" in str(hypothesis_text or "").lower():
        return "required"
    if analysis_type in {"correlation", "regression", "survival"}:
        return "recommended"
    return "none"


def infer_analysis_type(plan: dict, hypothesis: str) -> str:
    """Infer analysis_type if not specified in plan."""
    if plan.get("analysis_type"):
        return plan["analysis_type"]

    text = (hypothesis or "").lower()
    if "correlat" in text or "association" in text:
        return "correlation"
    if "regress" in text or "predict" in text:
        return "regression"

    return "group_difference"


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _build_metadata_alias_lookup(metadata_fields: list[str]) -> dict[str, str]:
    alias_to_field: dict[str, str] = {}
    for field in metadata_fields:
        canonical = str(field)
        token = _normalize_token(canonical)
        if token:
            alias_to_field[token] = canonical

    manual_aliases = {
        "ed frame": "ed_frame",
        "ed frame index": "ed_frame",
        "end diastolic frame": "ed_frame",
        "end diastole frame": "ed_frame",
        "es frame": "es_frame",
        "es frame index": "es_frame",
        "end systolic frame": "es_frame",
        "end systole frame": "es_frame",
        "num frames": "num_frames",
        "number of frames": "num_frames",
        "number of cine frames": "num_frames",
        "cine frames": "num_frames",
        "frame count": "num_frames",
        "body weight": "weight",
        "body height": "height",
    }
    available = {str(field) for field in metadata_fields}
    for alias, field in manual_aliases.items():
        if field in available:
            alias_to_field[_normalize_token(alias)] = field
    return alias_to_field


def _resolve_metadata_field(value: str, metadata_fields: list[str]) -> Optional[str]:
    if not str(value or "").strip():
        return None
    lookup = _build_metadata_alias_lookup(metadata_fields)
    return lookup.get(_normalize_token(value))


def _extract_hypothesis_metadata_fields(hypothesis: str, metadata_fields: list[str]) -> list[str]:
    text = _normalize_token(hypothesis or "")
    if not text:
        return []
    alias_lookup = _build_metadata_alias_lookup(metadata_fields)
    mentions: list[str] = []
    for alias, field in alias_lookup.items():
        if re.search(rf"\b{re.escape(alias)}\b", text) and field not in mentions:
            mentions.append(field)
    return mentions


def _detect_metadata_group_field(
    group_values: list[str],
    metadata_fields: list[str],
    domain_notes: str,
) -> Optional[str]:
    """Detect which metadata field contains the group values.

    Parses domain_notes for explicit field references like
    "Filter by metadata field idh_status" or matches group value
    substrings against field names.
    """
    import re

    if not group_values or not metadata_fields:
        return None

    # 1. Parse domain_notes for explicit "metadata field <name>" pattern
    if domain_notes:
        pattern = r"(?:metadata\s+field|filter\s+by)\s+['\"]?(\w+)['\"]?"
        match = re.search(pattern, domain_notes, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            # Verify it's actually a metadata field
            for field in metadata_fields:
                if field.lower() == candidate.lower():
                    return field

    # 2. Heuristic: check if any group value is a substring of a field name
    #    e.g., groups=['wildtype','mutant'] might match field 'idh_status'
    #    This is weak, so we only use it if there's a single matching field
    group_lower = {str(v).strip().lower() for v in group_values}
    for field in metadata_fields:
        field_lower = field.lower().replace("_", " ")
        for gv in group_lower:
            if gv in field_lower:
                return field

    return None


def normalize_target_variables(
    target_variables,
    metadata_fields: list[str],
) -> dict:
    """Normalize Phase 1 target_variables contract for Phase 2B validation.

    Strict mode: normalize only explicitly provided target variables; do not infer
    missing outcome/predictors from other fields.
    """
    target = target_variables if isinstance(target_variables, dict) else {}

    raw_predictors = target.get("predictors")
    if isinstance(raw_predictors, str):
        raw_predictors = [raw_predictors]
    if not isinstance(raw_predictors, list):
        raw_predictors = []

    normalized_predictors: list[str] = []
    seen_predictors = set()
    for predictor in raw_predictors:
        value = str(predictor).strip()
        if not value:
            continue
        resolved = _resolve_metadata_field(value, metadata_fields)
        canonical = resolved or value
        if canonical not in seen_predictors:
            normalized_predictors.append(canonical)
            seen_predictors.add(canonical)

    outcome = target.get("outcome")
    if isinstance(outcome, list):
        outcome = outcome[0] if outcome else None
    outcome_value = str(outcome).strip() if outcome is not None else ""
    if outcome_value:
        resolved_outcome = _resolve_metadata_field(outcome_value, metadata_fields)
        if resolved_outcome:
            outcome_value = resolved_outcome

    normalized = {
        "outcome": outcome_value,
        "predictors": normalized_predictors,
    }
    hypothesis_fields = target.get("hypothesis_metadata_fields")
    if isinstance(hypothesis_fields, list):
        normalized["hypothesis_metadata_fields"] = [
            str(field) for field in hypothesis_fields if str(field).strip()
        ]
    return normalized


def normalize_predictors(predictors, metadata_fields: list[str]) -> list[str]:
    """Normalize predictors to match available metadata fields."""
    if not predictors:
        return []
    if isinstance(predictors, str):
        predictors = [predictors]

    group_aliases = {"grade", "who grade", "tumor grade"}

    def normalize_field(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    all_valid_fields = list(metadata_fields) + ["group"]
    normalized = []
    seen = set()
    for pred in predictors:
        pred_norm = normalize_field(pred)
        resolved = None
        for field in all_valid_fields:
            if pred_norm == normalize_field(field):
                resolved = field
                break
        if resolved is None and pred_norm in group_aliases:
            resolved = "group"
        if resolved and resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)

    return normalized


def normalize_confounders(fields, metadata_fields: list[str]) -> list[str]:
    """Normalize adjust_for/stratify_by fields to match metadata fields.

    'group' is always valid (built-in attribute from sat.get_patient_metadata)
    even though it's not listed in patient_metadata_fields.
    """
    if not fields:
        return []
    if isinstance(fields, str):
        fields = [fields]

    group_aliases = {"grade", "who grade", "tumor grade"}

    # 'group' is always available from sat.get_patient_metadata()
    all_valid_fields = list(metadata_fields) + ["group"]

    def normalize_field(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    normalized = []
    seen = set()
    for item in fields:
        item_norm = normalize_field(item)
        resolved = None
        for field in all_valid_fields:
            if item_norm == normalize_field(field):
                resolved = field
                break
        if resolved is None and item_norm in group_aliases:
            resolved = "group"
        if resolved and resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)

    return normalized


if __name__ == "__main__":
    main()
