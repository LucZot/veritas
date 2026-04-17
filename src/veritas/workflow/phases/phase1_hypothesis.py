#!/usr/bin/env python3
"""Phase 1: Hypothesis Planning

Agents collaborate to create a concrete analysis plan.
"""

import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EXECUTION_MODE", "langgraph")
os.environ.setdefault("OLLAMA_CONTEXT_LENGTH", os.environ.get("VERITAS_CONTEXT_LENGTH", "16384"))

repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from veritas import run_meeting
from veritas.prompts import (
    PRINCIPAL_INVESTIGATOR,
    PHASE_AWARE_CRITIC,
    MEDICAL_IMAGING_SPECIALIST_DISCUSSION,
    ML_STATISTICIAN_DISCUSSION,
    create_agent_with_model,
)
from veritas.utils import archive_meeting_artifacts, load_summaries
from veritas.utils.discussion import strip_thinking_process
from veritas.verbosity import get_prompt_verbosity_config
from veritas.prompt_templates import build_phase1_agenda, build_phase1_summary_instructions
from veritas.workflow.audit import (
    append_correction_report,
    classify_failure_code,
    update_workflow_audit,
    write_validation_report,
)

UNTESTABLE_SUBTYPES = {
    "UNTESTABLE_MISSING_STRUCTURE",
    "UNTESTABLE_MISSING_METADATA_FIELD",
    "UNTESTABLE_MISSING_MODALITY",
    "UNTESTABLE_MISSING_MEASUREMENT",
    "UNTESTABLE_OTHER",
}

@dataclass
class PhaseConfig:
    """Configuration for phase execution."""
    output_path: str = "outputs/workflow"
    pi_model: str = "gpt-oss:20b"
    imaging_discussion_model: str = "gpt-oss:20b"
    statistician_discussion_model: str = "qwen3:8b"
    temperature: float = 0.2
    top_p: Optional[float] = None
    # Per-role temperature overrides (None = use global temperature)
    pi_temperature: Optional[float] = None
    imaging_discussion_temperature: Optional[float] = None
    statistician_discussion_temperature: Optional[float] = None
    # Per-role top_p overrides (None = use global top_p)
    pi_top_p: Optional[float] = None
    imaging_discussion_top_p: Optional[float] = None
    statistician_discussion_top_p: Optional[float] = None
    prompt_verbosity: str = "standard"
    hypothesis: str = ""
    dataset_path: str = ""
    dataset_metadata: Optional[dict] = None
    critic_model: str = "qwen3:8b"
    enable_critic_phase1: bool = False


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def load_config() -> PhaseConfig:
    """Load configuration from config file or environment variables."""
    config_file = os.environ.get("WORKFLOW_CONFIG_FILE")

    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            data = json.load(f)

        # Get models from nested structure or flat structure
        models = data.get("models") or {}
        temperatures = data.get("temperatures") or {}
        top_ps = data.get("top_ps") or {}

        return PhaseConfig(
            output_path=data.get("output_path", PhaseConfig.output_path),
            pi_model=models.get("pi", data.get("pi_model", PhaseConfig.pi_model)),
            imaging_discussion_model=models.get("imaging_discussion", data.get("imaging_discussion_model", PhaseConfig.imaging_discussion_model)),
            statistician_discussion_model=models.get("statistician_discussion", data.get("statistician_discussion_model", PhaseConfig.statistician_discussion_model)),
            temperature=data.get("temperature", PhaseConfig.temperature),
            top_p=data.get("top_p"),
            pi_temperature=temperatures.get("pi"),
            imaging_discussion_temperature=temperatures.get("imaging_discussion"),
            statistician_discussion_temperature=temperatures.get("statistician_discussion"),
            pi_top_p=top_ps.get("pi"),
            imaging_discussion_top_p=top_ps.get("imaging_discussion"),
            statistician_discussion_top_p=top_ps.get("statistician_discussion"),
            prompt_verbosity=data.get("prompt_verbosity", PhaseConfig.prompt_verbosity),
            hypothesis=data.get("hypothesis", ""),
            dataset_path=data.get("dataset_path", ""),
            dataset_metadata=data.get("dataset"),
            critic_model=models.get("critic", data.get("critic_model", PhaseConfig.critic_model)),
            enable_critic_phase1=_parse_bool(
                data.get("enable_critic_phase1"),
                default=PhaseConfig.enable_critic_phase1,
            ),
        )

    # Fall back to environment variables
    metadata_str = os.environ.get("DATASET_METADATA")
    dataset_metadata = json.loads(metadata_str) if metadata_str else None

    return PhaseConfig(
        output_path=os.environ.get("WORKFLOW_OUTPUT_PATH", PhaseConfig.output_path),
        pi_model=os.environ.get("PI_MODEL", PhaseConfig.pi_model),
        imaging_discussion_model=os.environ.get("IMAGING_DISCUSSION_MODEL", PhaseConfig.imaging_discussion_model),
        statistician_discussion_model=os.environ.get("STATISTICIAN_DISCUSSION_MODEL", PhaseConfig.statistician_discussion_model),
        temperature=float(os.environ.get("WORKFLOW_TEMPERATURE", str(PhaseConfig.temperature))),
        prompt_verbosity=os.environ.get("WORKFLOW_PROMPT_VERBOSITY", PhaseConfig.prompt_verbosity),
        hypothesis=os.environ.get("HYPOTHESIS", ""),
        dataset_path=os.environ.get("DATASET_PATH", ""),
        dataset_metadata=dataset_metadata,
        critic_model=os.environ.get("CRITIC_MODEL", PhaseConfig.critic_model),
        enable_critic_phase1=_parse_bool(
            os.environ.get("WORKFLOW_ENABLE_CRITIC_PHASE1"),
            default=PhaseConfig.enable_critic_phase1,
        ),
    )


def format_dataset_info(metadata: dict) -> str:
    """Format dataset metadata for agenda."""
    if not metadata:
        return "**Dataset:** No metadata provided"

    lines = ["**Dataset:**"]

    if "name" in metadata:
        lines.append(f"- Name: {metadata['name']}")
    if "domain" in metadata:
        lines.append(f"- Domain: {metadata['domain']}")
    if "modality" in metadata:
        lines.append(f"- Modality: {metadata['modality']}")
    if "available_groups" in metadata:
        groups = ", ".join(metadata["available_groups"])
        lines.append(f"- Patient groups: {groups}")
    if "available_observations" in metadata:
        obs = ", ".join(metadata["available_observations"])
        lines.append(f"- Observations: {obs}")
    if "patient_metadata_fields" in metadata:
        fields = ", ".join(metadata["patient_metadata_fields"])
        lines.append(f"- Patient metadata: {fields}")
    if "domain_notes" in metadata:
        lines.append(f"\n{metadata['domain_notes']}")

    return "\n".join(lines)


def main():
    """Run Phase 1: Hypothesis planning."""
    print("\n" + "=" * 70)
    print("PHASE 1: HYPOTHESIS PLANNING")
    print("=" * 70 + "\n")

    config = load_config()

    if not config.hypothesis:
        raise ValueError("No hypothesis provided in config")

    # Resolve output path
    output_path = Path(config.output_path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    save_dir = (output_path / "phase1_hypothesis_formulation").absolute()
    save_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = save_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CODE_EXEC_WORKSPACE_DIR'] = str(workspace_dir)

    config_path = os.environ.get("WORKFLOW_CONFIG_FILE")
    if not config_path:
        raise RuntimeError("WORKFLOW_CONFIG_FILE not set")
    config_path_obj = Path(config_path)
    update_workflow_audit(
        config_file=config_path_obj,
        phase="phase1",
        status="running",
        details={"script": "phase1_hypothesis.py"},
    )
    # Format dataset info
    dataset_info_text = format_dataset_info(config.dataset_metadata)
    print("📊 Dataset Information:")
    print(dataset_info_text)
    print()

    # Create agents
    pi = create_agent_with_model(PRINCIPAL_INVESTIGATOR, config.pi_model, config.pi_temperature, config.pi_top_p)
    imaging_specialist = create_agent_with_model(
        MEDICAL_IMAGING_SPECIALIST_DISCUSSION,
        config.imaging_discussion_model,
        config.imaging_discussion_temperature,
        config.imaging_discussion_top_p,
    )
    statistician = create_agent_with_model(
        ML_STATISTICIAN_DISCUSSION,
        config.statistician_discussion_model,
        config.statistician_discussion_temperature,
        config.statistician_discussion_top_p,
    )
    critic = create_agent_with_model(PHASE_AWARE_CRITIC, config.critic_model)

    # Build agenda and summary instructions from templates
    agenda = build_phase1_agenda(
        hypothesis=config.hypothesis,
        dataset_info_text=dataset_info_text,
        dataset_metadata=config.dataset_metadata,
    )
    summary_instructions = build_phase1_summary_instructions(
        dataset_metadata=config.dataset_metadata,
    )

    print(f"Running team meeting with:")
    print(f"  • PI: {config.pi_model}")
    print(f"  • Imaging Specialist: {config.imaging_discussion_model}")
    print(f"  • Statistician: {config.statistician_discussion_model}")
    print(f"  • Critic enabled: {config.enable_critic_phase1}")
    print(f"  • Temperature: {config.temperature}")
    print(f"  • Verbosity: {config.prompt_verbosity}")
    print()

    run_meeting(
        meeting_type="team",
        team_lead=pi,
        team_members=(imaging_specialist, statistician),
        agenda=agenda,
        save_dir=save_dir,
        critic=critic,
        num_rounds=1,
        temperature=config.temperature,
        top_p=config.top_p,
        prompt_verbosity=config.prompt_verbosity,
        enable_critic=config.enable_critic_phase1,
        workflow_instruction="Phase 1: Hypothesis Planning - Design analysis plan",
        summary_instructions=summary_instructions,
    )

    print(f"\n✓ Phase 1 complete. Results saved to {save_dir}")
    print(f"  • Discussion: {save_dir / 'discussion.json'}")
    print(f"  • Summary: {save_dir / 'summary.md'}")

    plan = None
    contract_repairs = []
    correction_attempted = False
    correction_success = None
    try:
        summary_text = load_summary(save_dir)
        plan = extract_plan_block(summary_text)

        if not plan:
            raise ValueError(
                "Plan JSON not found in PI summary. "
                "Phase 1 requires a valid plan with all 5 keys: "
                "groups, structures, observations, metrics, statistical_test"
            )

        plan, contract_repairs = normalize_plan_contract(plan)
        write_plan_contract_audit(save_dir, contract_repairs)

        plan = normalize_feasibility(plan)
        if is_plan_untestable(plan):
            feasibility = plan.get("feasibility", {})
            print("  ⚠ Phase 1 marked hypothesis as UNTESTABLE")
            print(f"  • Invalid subtype: {feasibility.get('invalid_subtype', 'UNTESTABLE_OTHER')}")
            if feasibility.get("reason"):
                print(f"  • Reason: {feasibility['reason']}")
            if feasibility.get("missing_requirements"):
                print(f"  • Missing requirements: {feasibility['missing_requirements']}")
            plan_artifact_path = write_final_plan_artifacts(
                save_dir=save_dir,
                plan=plan,
                source_discussion="discussion.json",
                correction_attempted=False,
                correction_success=None,
            )
            update_config_plan(config_path_obj, plan, plan_artifact_path=plan_artifact_path)
            write_validation_report(
                save_dir=save_dir,
                phase="phase1",
                status="passed",
                checks={
                    "plan_extracted": True,
                    "contract_repair_count": len(contract_repairs),
                    "untestable": True,
                },
                warnings=["phase1_marked_untestable"],
            )
            update_workflow_audit(
                config_file=config_path_obj,
                phase="phase1",
                status="passed",
                details={"untestable": True, "contract_repair_count": len(contract_repairs)},
            )
            print(f"  • Plan artifact: {plan_artifact_path}")
            print(f"  • Plan saved to config: {config_path}")
            return

        missing_required = [
            key
            for key in ("groups", "structures", "observations", "metrics", "statistical_test")
            if key not in plan
        ]
        if missing_required:
            raise ValueError(
                "Plan JSON is missing required keys after contract normalization: "
                + ", ".join(missing_required)
            )

        # Minimal formatting normalization (logged, not semantic)
        plan = normalize_observations(plan, config.dataset_metadata)
        plan = normalize_cohort_mode(plan)
        plan = normalize_grouping_field(plan, config.dataset_metadata)
        plan = normalize_target_variables(plan, config.hypothesis, config.dataset_metadata)
        if plan.get("observations"):
            print(f"  • Observations validated: {plan['observations']}")
        if plan.get("cohort_mode"):
            print(f"  • Cohort mode: {plan['cohort_mode']}")

        # Validate plan against dataset metadata
        issues = validate_plan(
            plan,
            config.hypothesis,
            config.dataset_metadata,
        )
        if issues:
            correction_attempted = True
            print(f"\n  ⚠ Plan validation found {len(issues)} issue(s):")
            for issue in issues:
                print(f"    - {issue}")
            print("  → Requesting PI to correct the plan...")

            corrected_plan = request_plan_correction(
                save_dir=save_dir,
                correction_index=1,
                pi=pi,
                plan=plan,
                issues=issues,
                dataset_metadata=config.dataset_metadata,
                hypothesis=config.hypothesis,
                temperature=config.temperature,
                top_p=config.top_p,
                prompt_verbosity=config.prompt_verbosity,
            )
            if corrected_plan:
                plan = corrected_plan
                plan = normalize_observations(plan, config.dataset_metadata)
                plan = normalize_cohort_mode(plan)
                plan = normalize_grouping_field(plan, config.dataset_metadata)
                plan = normalize_target_variables(plan, config.hypothesis, config.dataset_metadata)
                # Re-validate after correction
                remaining_issues = validate_plan(
                    plan,
                    config.hypothesis,
                    config.dataset_metadata,
                )
                if remaining_issues:
                    correction_success = False
                    append_correction_report(
                        save_dir=save_dir,
                        phase="phase1",
                        attempted=True,
                        success=False,
                        reason="pi_correction_failed_validation",
                        error_before="; ".join(issues),
                        error_after="; ".join(remaining_issues),
                    )
                    print(f"  ⚠ PI correction still has {len(remaining_issues)} issue(s):")
                    for issue in remaining_issues:
                        print(f"    - {issue}")
                    raise ValueError(
                        "Plan validation failed after PI correction. Issues: "
                        + "; ".join(remaining_issues)
                    )
                correction_success = True
                append_correction_report(
                    save_dir=save_dir,
                    phase="phase1",
                    attempted=True,
                    success=True,
                    reason="pi_correction_succeeded",
                    error_before="; ".join(issues),
                )
                print("  ✓ PI corrected the plan successfully")
            else:
                correction_success = False
                append_correction_report(
                    save_dir=save_dir,
                    phase="phase1",
                    attempted=True,
                    success=False,
                    reason="pi_correction_no_valid_json",
                    error_before="; ".join(issues),
                )
                raise ValueError(
                    "Plan validation failed and PI could not produce a corrected plan. "
                    "Issues: " + "; ".join(issues)
                )

        final_plan_source = "discussion_correction_01.json" if correction_success else "discussion.json"
        plan_artifact_path = write_final_plan_artifacts(
            save_dir=save_dir,
            plan=plan,
            source_discussion=final_plan_source,
            correction_attempted=correction_attempted,
            correction_success=correction_success,
        )
        update_config_plan(config_path_obj, plan, plan_artifact_path=plan_artifact_path)
        write_validation_report(
            save_dir=save_dir,
            phase="phase1",
            status="passed",
            checks={
                "plan_extracted": True,
                "contract_repair_count": len(contract_repairs),
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase1",
            status="passed",
            details={
                "contract_repair_count": len(contract_repairs),
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        print(f"  • Plan saved to config: {config_path}")
    except Exception as e:
        failure_code = classify_failure_code("phase1", str(e))
        write_validation_report(
            save_dir=save_dir,
            phase="phase1",
            status="failed",
            checks={
                "plan_extracted": bool(plan),
                "contract_repair_count": len(contract_repairs),
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
            errors=[str(e)],
            failure_code=failure_code,
        )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase1",
            status="failed",
            failure_code=failure_code,
            failure_message=str(e),
            details={
                "contract_repair_count": len(contract_repairs),
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        raise


def load_summary(save_dir: Path) -> str:
    """Load and clean summary text."""
    summary_path = save_dir / "summary.md"
    if summary_path.exists():
        text = summary_path.read_text()
        cleaned = strip_thinking_process(text)
        if cleaned != text:
            summary_path.write_text(cleaned)
        return cleaned

    # Fall back to last message in discussion.json
    discussion_path = save_dir / "discussion.json"
    if not discussion_path.exists():
        raise FileNotFoundError(f"No summary or discussion found in {save_dir}")

    with open(discussion_path) as f:
        discussion = json.load(f)

    if not discussion:
        raise ValueError("Discussion is empty")

    text = discussion[-1].get("message", "")
    if not text:
        raise ValueError("Last discussion message is empty")

    cleaned = strip_thinking_process(text)
    summary_path.write_text(cleaned)
    return cleaned


def extract_plan_block(text: str) -> dict:
    """Extract plan JSON from text.

    Returns the LAST valid JSON block with all required keys.
    Allows UNTESTABLE feasibility plans to omit standard plan keys.
    """
    required_keys = [
        "groups",
        "structures",
        "observations",
        "metrics",
        "statistical_test",
    ]

    # Try fenced JSON blocks first
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]

    last_valid = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                plan = json.loads(match.group(1))
                missing = [k for k in required_keys if k not in plan]
                if not missing or _candidate_marks_untestable(plan):
                    last_valid = plan
            except json.JSONDecodeError:
                continue

    if last_valid:
        return last_valid

    # Try balanced JSON objects (handles nested dicts without fences)
    for candidate in _iter_balanced_json_objects(text):
        try:
            plan = json.loads(candidate)
            missing = [k for k in required_keys if k not in plan]
            if not missing or _candidate_marks_untestable(plan):
                last_valid = plan
        except json.JSONDecodeError:
            continue

    return last_valid


def _iter_balanced_json_objects(text: str):
    """Yield balanced {...} substrings from text.

    This is robust to nested JSON objects and quoted braces inside strings.
    """
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : idx + 1]
                    break


def _candidate_marks_untestable(plan: dict) -> bool:
    if not isinstance(plan, dict):
        return False
    feasibility = plan.get("feasibility")
    if not isinstance(feasibility, dict):
        return False
    status = str(feasibility.get("status") or "").strip().upper()
    return status == "UNTESTABLE"


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _build_metadata_alias_lookup(metadata_fields: list[str]) -> dict[str, str]:
    alias_to_field: dict[str, str] = {}
    for field in metadata_fields:
        canonical = str(field)
        norm = _normalize_token(canonical)
        if not norm:
            continue
        alias_to_field[norm] = canonical

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
    alias_lookup = _build_metadata_alias_lookup(metadata_fields)
    token = _normalize_token(value)
    if not token:
        return None
    if token in alias_lookup:
        return alias_lookup[token]
    return None


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


def normalize_plan_contract(plan: dict) -> tuple[dict, list[dict]]:
    """Normalize contract shape without changing scientific intent."""
    normalized = dict(plan or {})
    repairs: list[dict] = []

    def _record(field: str, action: str, before, after) -> None:
        repairs.append({
            "field": field,
            "action": action,
            "before": before,
            "after": after,
        })

    feasibility = normalized.get("feasibility") if isinstance(normalized.get("feasibility"), dict) else {}
    feasibility_status = str(feasibility.get("status") or "").strip().upper()
    is_untestable = feasibility_status == "UNTESTABLE"

    required_list_fields = ("groups", "structures", "observations", "metrics")
    optional_list_fields = ("predictors", "adjust_for", "stratify_by")
    for field in required_list_fields + optional_list_fields:
        value = normalized.get(field)
        if value is None:
            if field in optional_list_fields or is_untestable:
                normalized[field] = []
                _record(field, "default_empty_list", None, [])
            continue
        if isinstance(value, str):
            normalized[field] = [value]
            _record(field, "coerce_str_to_list", value, [value])
            continue
        if not isinstance(value, list):
            coerced = [str(value)]
            normalized[field] = coerced
            _record(field, "coerce_scalar_to_list", value, coerced)

    restrict_to = normalized.get("restrict_to")
    if restrict_to is None:
        normalized["restrict_to"] = {}
        _record("restrict_to", "default_empty_dict", None, {})
    elif not isinstance(restrict_to, dict):
        normalized["restrict_to"] = {}
        _record("restrict_to", "coerce_invalid_to_empty_dict", restrict_to, {})

    statistical_test = normalized.get("statistical_test")
    if isinstance(statistical_test, list) and statistical_test:
        normalized["statistical_test"] = str(statistical_test[0])
        _record("statistical_test", "coerce_list_to_first_item", statistical_test, normalized["statistical_test"])
    elif statistical_test is not None and not isinstance(statistical_test, str):
        normalized["statistical_test"] = str(statistical_test)
        _record("statistical_test", "coerce_non_str_to_str", statistical_test, normalized["statistical_test"])

    if "grouping_field" in normalized:
        grouping_field = normalized.get("grouping_field")
        if grouping_field is not None and not isinstance(grouping_field, str):
            normalized["grouping_field"] = str(grouping_field)
            _record("grouping_field", "coerce_non_str_to_str", grouping_field, normalized["grouping_field"])

    group_spec = normalized.get("group_spec")
    if group_spec is not None and not isinstance(group_spec, dict):
        normalized["group_spec"] = None
        _record("group_spec", "drop_invalid_group_spec", group_spec, None)

    target_variables = normalized.get("target_variables")
    if target_variables is not None and not isinstance(target_variables, dict):
        normalized["target_variables"] = {}
        _record("target_variables", "coerce_invalid_to_empty_dict", target_variables, {})

    return normalized, repairs


def write_plan_contract_audit(save_dir: Path, repairs: list[dict]) -> None:
    """Persist contract-repair provenance for auditability."""
    audit_path = save_dir / "plan_contract_audit.json"
    payload = {
        "repair_count": len(repairs),
        "repairs": repairs,
    }
    with open(audit_path, "w") as f:
        json.dump(payload, f, indent=2)


def normalize_feasibility(plan: dict) -> dict:
    """Ensure feasibility block exists and is normalized."""
    feasibility = plan.get("feasibility")
    if not isinstance(feasibility, dict):
        plan["feasibility"] = {
            "status": "TESTABLE",
            "invalid_subtype": None,
            "reason": None,
            "missing_requirements": [],
        }
        return plan

    status = str(feasibility.get("status") or "TESTABLE").strip().upper()
    if status not in {"TESTABLE", "UNTESTABLE"}:
        status = "TESTABLE"

    invalid_subtype = feasibility.get("invalid_subtype")
    if status == "UNTESTABLE":
        invalid_subtype = str(invalid_subtype or "UNTESTABLE_OTHER").strip().upper()
        if invalid_subtype not in UNTESTABLE_SUBTYPES:
            invalid_subtype = "UNTESTABLE_OTHER"
    else:
        invalid_subtype = None

    missing_requirements = feasibility.get("missing_requirements")
    if isinstance(missing_requirements, str):
        missing_requirements = [missing_requirements]
    if not isinstance(missing_requirements, list):
        missing_requirements = []
    missing_requirements = [str(item) for item in missing_requirements if str(item).strip()]

    reason = feasibility.get("reason")
    if reason is not None:
        reason = str(reason)

    plan["feasibility"] = {
        "status": status,
        "invalid_subtype": invalid_subtype,
        "reason": reason,
        "missing_requirements": missing_requirements,
    }
    return plan


def is_plan_untestable(plan: dict) -> bool:
    feasibility = plan.get("feasibility") if isinstance(plan, dict) else None
    if not isinstance(feasibility, dict):
        return False
    return str(feasibility.get("status") or "").strip().upper() == "UNTESTABLE"


def validate_plan(
    plan: dict,
    hypothesis: str,
    dataset_metadata: Optional[dict],
) -> list:
    """Validate plan against dataset metadata.

    Returns a list of issue strings. Empty list means plan is valid.
    Checks: predictor validity, confounder validity, group adjustment requirement.
    """
    if not plan or not dataset_metadata:
        return []

    issues = []
    metadata_fields = dataset_metadata.get("patient_metadata_fields", []) or []
    available_groups = dataset_metadata.get("available_groups", []) or []
    available_observations = dataset_metadata.get("available_observations", []) or []
    analysis_type = (plan.get("analysis_type") or "group_difference").lower()

    def normalize_field(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    hypothesis_text = normalize_field(hypothesis or "")

    field_lookup = {normalize_field(f): f for f in metadata_fields}
    group_aliases = {"grade", "who grade", "tumor grade"}

    def resolve_field_name(field_name: str) -> Optional[str]:
        normalized = normalize_field(field_name)
        if normalized == "group":
            return "group"
        if normalized in field_lookup:
            return field_lookup[normalized]
        if normalized in group_aliases:
            return "group"
        return None

    # 0. Check observations against dataset available_observations
    if available_observations:
        obs_lookup = {normalize_field(o): o for o in available_observations}
        plan_obs = plan.get("observations") or []
        if isinstance(plan_obs, str):
            plan_obs = [plan_obs]
        invalid_obs = [o for o in plan_obs if normalize_field(o) not in obs_lookup]
        if invalid_obs:
            metadata_like_obs = [o for o in invalid_obs if normalize_field(o) in field_lookup]
            if metadata_like_obs:
                issues.append(
                    "Observations must be imaging observations/timepoints only. "
                    f"These entries look like metadata/clinical fields and must be moved out of observations: "
                    f"{metadata_like_obs}. Available imaging observations: {available_observations}"
                )
            else:
                issues.append(
                    f"Observations {invalid_obs} are not in available observations: {available_observations}"
                )

    # 0b. Segmentation-based plans require observations
    structures = plan.get("structures") or []
    if isinstance(structures, str):
        structures = [structures]
    plan_obs = plan.get("observations") or []
    if isinstance(plan_obs, str):
        plan_obs = [plan_obs]
    if structures and len(plan_obs) == 0:
        issues.append(
            "Plans with non-empty structures require non-empty observations. "
            "Provide observation/timepoint labels from available observations."
        )
    metrics = plan.get("metrics") or []
    if isinstance(metrics, str):
        metrics = [metrics]
    if structures and len(metrics) == 0:
        issues.append(
            "Plans with non-empty structures require non-empty metrics. "
            "Specify at least one imaging-derived metric (e.g., volume, mass, fraction, ratio)."
        )

    # 1. Check predictors for correlation/regression
    if analysis_type in {"correlation", "regression"}:
        predictors = plan.get("predictors") or []
        if isinstance(predictors, str):
            predictors = [predictors]

        if not predictors:
            issues.append(
                f"Predictors are required for {analysis_type} analysis but none were specified. "
                f"Available metadata fields: {metadata_fields}"
            )
        else:
            for pred in predictors:
                resolved_pred = resolve_field_name(pred)
                if not resolved_pred:
                    issues.append(
                        f"Predictor '{pred}' is not a patient metadata field. "
                        f"Available: {metadata_fields + ['group']} (grade aliases map to 'group')."
                    )

        target_variables = plan.get("target_variables")
        if not isinstance(target_variables, dict):
            issues.append(
                "target_variables is required for correlation/regression and must be an object "
                "with keys: outcome, predictors."
            )
        else:
            target_outcome = target_variables.get("outcome")
            target_predictors = target_variables.get("predictors")
            if isinstance(target_predictors, str):
                target_predictors = [target_predictors]
            if not isinstance(target_predictors, list):
                target_predictors = []

            if not str(target_outcome or "").strip():
                issues.append("target_variables.outcome must be a non-empty string.")
            if len(target_predictors) == 0:
                issues.append("target_variables.predictors must contain at least one variable.")

            normalized_plan_predictors = {
                normalize_field(resolve_field_name(pred) or pred)
                for pred in predictors
                if str(pred).strip()
            }
            normalized_target_predictors = {
                normalize_field(resolve_field_name(pred) or pred)
                for pred in target_predictors
                if str(pred).strip()
            }
            if normalized_plan_predictors and normalized_target_predictors and normalized_plan_predictors != normalized_target_predictors:
                issues.append(
                    "target_variables.predictors must match predictors in the plan for correlation/regression analyses."
                )

            hypothesis_fields = _extract_hypothesis_metadata_fields(hypothesis, metadata_fields)
            if hypothesis_fields:
                resolved_target_outcome = resolve_field_name(target_outcome) if target_outcome else None
                resolved_target_predictors = [
                    resolve_field_name(pred) for pred in target_predictors if str(pred).strip()
                ]
                variable_pool = {
                    field
                    for field in ([resolved_target_outcome] + resolved_target_predictors)
                    if field
                }
                missing_hypothesis_fields = [
                    field for field in hypothesis_fields
                    if field not in variable_pool
                ]
                if missing_hypothesis_fields:
                    issues.append(
                        "target_variables does not match metadata variables named in the hypothesis. "
                        f"Missing: {missing_hypothesis_fields}; detected from hypothesis: {hypothesis_fields}."
                    )

    # 2. Check confounders (adjust_for, stratify_by)
    SPECIAL_FIELDS = {"group"}
    for key in ("adjust_for", "stratify_by"):
        raw = plan.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        for item in raw:
            resolved_item = resolve_field_name(item)
            if resolved_item in SPECIAL_FIELDS:
                continue
            if not resolved_item:
                issues.append(
                    f"{key} field '{item}' is not a patient metadata field or 'group'. "
                    f"Available: {metadata_fields + ['group']} (grade aliases map to 'group')."
                )

    # 3a. Check survival analysis type when metrics include survival outcomes
    SURVIVAL_METRICS = {"survival_days", "survival", "overall_survival", "os", "pfs",
                        "progression_free_survival"}
    metrics = plan.get("metrics") or []
    if isinstance(metrics, str):
        metrics = [metrics]
    has_survival_metric = any(
        normalize_field(m).replace(" ", "_") in SURVIVAL_METRICS for m in metrics
    )
    if has_survival_metric and analysis_type != "survival":
        issues.append(
            f"Metrics include a survival outcome ({metrics}) but analysis_type is "
            f"'{analysis_type}'. Survival outcomes require analysis_type: 'survival'."
        )
    if has_survival_metric and analysis_type == "survival":
        test_name = str(plan.get("statistical_test") or "").strip().lower()
        if not any(token in test_name for token in ("log-rank", "logrank", "cox")):
            issues.append(
                "Survival analysis requires a survival test: use log-rank for unadjusted "
                "group comparisons or Cox PH for adjusted/continuous survival modeling."
            )
        adjust_for_fields = plan.get("adjust_for") or []
        if isinstance(adjust_for_fields, str):
            adjust_for_fields = [adjust_for_fields]
        if adjust_for_fields and "cox" not in test_name:
            issues.append(
                "Survival plan includes adjust_for covariates but statistical_test is not Cox PH. "
                "Use Cox proportional hazards model when adjustment is required."
            )

    # 3b. Metadata/derived groups must declare grouping source
    groups = plan.get("groups") or []
    available_groups_set = {str(g).strip() for g in available_groups}
    metadata_based_groups = [
        g for g in groups if str(g).strip() not in available_groups_set and str(g).strip().lower() != "all"
    ]
    group_spec = plan.get("group_spec") if isinstance(plan.get("group_spec"), dict) else None
    group_spec_type = str((group_spec or {}).get("type") or "").strip().lower()

    grouping_field = plan.get("grouping_field")
    if not grouping_field and group_spec_type == "metadata":
        grouping_field = (group_spec or {}).get("field")

    if metadata_based_groups:
        looks_derived = _looks_like_derived_groups(metadata_based_groups)
        if group_spec_type == "derived" or looks_derived:
            pass
        elif grouping_field and str(grouping_field).strip():
            resolved_grouping = resolve_field_name(grouping_field)
            if not resolved_grouping:
                issues.append(
                    f"grouping_field '{grouping_field}' is not a patient metadata field. "
                    f"Available: {metadata_fields + ['group']} (grade aliases map to 'group')."
                )
        elif group_spec_type == "metadata":
            issues.append(
                "group_spec.type='metadata' requires a valid grouping_field that points to a "
                f"patient metadata field for values {metadata_based_groups}. "
                f"Available metadata fields: {metadata_fields}"
            )

    # 3c. Group-difference analyses require at least two groups
    if analysis_type == "group_difference":
        valid_groups = [g for g in groups if str(g).strip().lower() != "all"]
        if len(valid_groups) < 2:
            issues.append(
                "Group-difference analysis requires at least two comparison groups "
                "(or metadata values via grouping_field)."
            )

    # 3d. Prevent over-escalating direct group comparisons into regression/correlation
    if analysis_type in {"correlation", "regression"}:
        predictors = plan.get("predictors") or []
        if isinstance(predictors, str):
            predictors = [predictors]
        resolved_predictors = [resolve_field_name(pred) for pred in predictors]
        only_group_predictor = bool(resolved_predictors) and all(pred == "group" for pred in resolved_predictors if pred)
        explicit_association = any(
            token in hypothesis_text
            for token in (
                "correlation", "correlat", "association", "associated", "regression",
                "predict", "relationship", "linked"
            )
        )
        if only_group_predictor and not explicit_association:
            issues.append(
                "This appears to be a direct group comparison hypothesis. "
                "Use analysis_type='group_difference' instead of regression/correlation."
            )

    # 3e. If hypothesis explicitly requests adjustment, require adjust_for
    explicit_adjustment = any(
        token in hypothesis_text
        for token in (
            "adjusting for",
            "adjusted for",
            "when accounting for",
            "accounting for",
            "controlling for",
            "after adjusting",
        )
    )
    if explicit_adjustment and analysis_type in {"correlation", "regression", "survival"}:
        adjust_for = plan.get("adjust_for") or []
        if isinstance(adjust_for, str):
            adjust_for = [adjust_for]
        if len(adjust_for) == 0:
            issues.append(
                "Hypothesis explicitly requests adjusted analysis, but adjust_for is empty. "
                "Specify covariates in adjust_for."
            )

    return issues


def _looks_like_derived_groups(groups: list) -> bool:
    tokens = {str(group).strip().lower() for group in groups if str(group).strip()}
    if not tokens:
        return False
    derived_markers = (
        "high",
        "low",
        "upper",
        "lower",
        "q1",
        "q2",
        "q3",
        "q4",
        "quartile",
        "tertile",
        "threshold",
        "above",
        "below",
        "median",
    )
    return all(any(marker in token for marker in derived_markers) for token in tokens)


def request_plan_correction(
    save_dir: Path,
    correction_index: int,
    pi: Any,
    plan: dict,
    issues: list,
    dataset_metadata: dict,
    hypothesis: str,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    prompt_verbosity: str = "standard",
) -> Optional[dict]:
    """Request a logged PI correction meeting for a Phase 1 plan.

    Runs a one-turn individual meeting so plan repair goes through the standard
    meeting machinery and leaves an artifact trail.
    """
    issues_text = "\n".join(f"  {i+1}. {issue}" for i, issue in enumerate(issues))

    metadata_fields = dataset_metadata.get("patient_metadata_fields", []) if dataset_metadata else []
    available_groups = dataset_metadata.get("available_groups", []) if dataset_metadata else []
    available_obs = dataset_metadata.get("available_observations", []) if dataset_metadata else []

    correction_agenda = (
        "Correct the Phase 1 plan using the validation feedback.\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Current plan:\n```json\n{json.dumps(plan, indent=2)}\n```\n\n"
        f"Validation issues:\n{issues_text}\n\n"
        "Dataset reference:\n"
        f"- groups: {available_groups}\n"
        f"- observations: {available_obs}\n"
        f"- metadata fields: {metadata_fields}\n\n"
        "Return one corrected JSON plan block (` ```json ... ``` `) only.\n"
        "Fix only listed issues and keep unrelated fields unchanged.\n"
        "For metadata-value groups, set `grouping_field`; for derived groups, set `group_spec.type='derived'`.\n"
        "Do not replace target quantity with proxies. If infeasible in principle, set feasibility to UNTESTABLE."
    )

    discussion_path = save_dir / "discussion.json"
    summaries = tuple()
    if discussion_path.exists():
        summaries = load_summaries((discussion_path,))

    save_name = f"discussion_correction_{correction_index:02d}"
    archive_meeting_artifacts(
        save_dir=save_dir,
        save_name=save_name,
        archive_name=f"{save_name}_previous",
    )

    try:
        correction_summary = run_meeting(
            meeting_type="individual",
            team_member=pi,
            agenda=correction_agenda,
            summaries=summaries,
            save_dir=save_dir,
            save_name=save_name,
            num_rounds=0,
            temperature=temperature,
            top_p=top_p,
            prompt_verbosity=prompt_verbosity,
            enable_critic=False,
            workflow_instruction="Phase 1: Plan Correction",
            return_summary=True,
        )
    except Exception as e:
        print(f"  ✗ PI correction meeting failed: {e}")
        return None

    response_text = strip_thinking_process(correction_summary or "")
    corrected = extract_plan_block(response_text)
    if not corrected:
        print("  ✗ PI correction meeting did not produce a valid plan JSON")
        return None

    correction_plan_path = save_dir / f"plan_correction_{correction_index:02d}.json"
    correction_plan_path.write_text(json.dumps(corrected, indent=2))
    print(f"  ✓ PI produced corrected plan ({correction_plan_path.name})")
    return corrected

def normalize_observations(plan: dict, dataset_metadata: Optional[dict]) -> dict:
    """Validate plan observations against dataset available_observations."""
    if not plan:
        return plan
    observations = plan.get("observations")
    if not observations:
        return plan
    if isinstance(observations, str):
        observations = [observations]

    available = dataset_metadata.get("available_observations") if dataset_metadata else None
    if not available:
        plan["observations"] = observations
        return plan

    def normalize_token(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    available_lower = {normalize_token(obs): obs for obs in available}

    normalized = []
    invalid = []
    for obs in observations:
        token = normalize_token(obs)
        if token in available_lower:
            normalized.append(available_lower[token])
        else:
            invalid.append(obs)

    if invalid:
        # Don't raise — invalid observations are reported by validate_plan() and
        # corrected via the PI feedback loop. Pass them through unchanged here.
        plan["observations"] = normalized + invalid
        return plan

    plan["observations"] = normalized
    return plan


def normalize_cohort_mode(plan: dict) -> dict:
    """Ensure cohort_mode is set consistently with groups."""
    if not plan:
        return plan
    cohort_mode = plan.get("cohort_mode")
    groups = plan.get("groups") or []
    group_tokens = [str(g).strip().lower() for g in groups]

    if cohort_mode:
        cohort_mode = str(cohort_mode).strip().lower()
        if cohort_mode in {"all", "groups"}:
            plan["cohort_mode"] = cohort_mode
            return plan

    if not groups or group_tokens == ["all"]:
        plan["cohort_mode"] = "all"
    else:
        plan["cohort_mode"] = "groups"
    return plan


def normalize_grouping_field(plan: dict, dataset_metadata: Optional[dict]) -> dict:
    """Normalize optional grouping_field to exact metadata field spelling."""
    if not plan:
        return plan

    grouping_field = plan.get("grouping_field")
    group_spec = plan.get("group_spec") if isinstance(plan.get("group_spec"), dict) else None
    if not grouping_field and group_spec and str(group_spec.get("type") or "").strip().lower() == "metadata":
        grouping_field = group_spec.get("field")
        plan["grouping_field"] = grouping_field

    if grouping_field is None:
        return plan
    if isinstance(grouping_field, str) and not grouping_field.strip():
        plan["grouping_field"] = None
        return plan

    metadata_fields = dataset_metadata.get("patient_metadata_fields", []) if dataset_metadata else []

    def normalize_token(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    grouping_norm = normalize_token(grouping_field)
    if grouping_norm == "group":
        plan["grouping_field"] = "group"
        return plan
    if grouping_norm in {"grade", "who grade", "tumor grade"}:
        plan["grouping_field"] = "group"
        return plan
    for field in metadata_fields:
        if grouping_norm == normalize_token(field):
            plan["grouping_field"] = field
            return plan

    # Keep unmatched value; validate_plan() will report a concrete issue
    return plan


def normalize_target_variables(
    plan: dict,
    hypothesis: str,
    dataset_metadata: Optional[dict],
) -> dict:
    """Normalize explicit target variable contract for correlation/regression analyses.

    Strict mode: normalize only what the agent explicitly provided; do not infer
    missing outcome/predictor variables from hypothesis text, metrics, or other fields.
    """
    if not plan:
        return plan

    analysis_type = str(plan.get("analysis_type") or "").strip().lower()
    if analysis_type not in {"correlation", "regression"}:
        return plan

    metadata_fields = dataset_metadata.get("patient_metadata_fields", []) if dataset_metadata else []
    target = plan.get("target_variables") if isinstance(plan.get("target_variables"), dict) else {}

    predictors = target.get("predictors")
    if isinstance(predictors, str):
        predictors = [predictors]
    if not isinstance(predictors, list):
        predictors = []

    normalized_predictors: list[str] = []
    seen = set()
    for predictor in predictors:
        predictor_str = str(predictor).strip()
        if not predictor_str:
            continue
        resolved = _resolve_metadata_field(predictor_str, metadata_fields)
        canonical = resolved or predictor_str
        if canonical not in seen:
            normalized_predictors.append(canonical)
            seen.add(canonical)

    outcome = target.get("outcome")
    if isinstance(outcome, list):
        outcome = outcome[0] if outcome else None
    outcome_str = str(outcome).strip() if outcome is not None else ""
    if outcome_str:
        resolved_outcome = _resolve_metadata_field(outcome_str, metadata_fields)
        if resolved_outcome:
            outcome_str = resolved_outcome

    mentions = _extract_hypothesis_metadata_fields(hypothesis, metadata_fields)

    normalized_target = {
        "outcome": outcome_str,
        "predictors": normalized_predictors,
    }
    if mentions:
        normalized_target["hypothesis_metadata_fields"] = mentions

    plan["target_variables"] = normalized_target
    return plan


def write_final_plan_artifacts(
    save_dir: Path,
    plan: dict,
    *,
    source_discussion: str,
    correction_attempted: bool,
    correction_success: Optional[bool],
) -> Path:
    """Persist the final accepted Phase 1 plan as a canonical artifact."""
    plan_path = save_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    provenance = {
        "plan_path": str(plan_path),
        "source_discussion": source_discussion,
        "correction_attempted": correction_attempted,
        "correction_success": correction_success,
    }
    correction_plan_path = save_dir / "plan_correction_01.json"
    if correction_plan_path.exists():
        provenance["correction_plan_path"] = str(correction_plan_path)

    provenance_path = save_dir / "plan_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    return plan_path


def update_config_plan(
    config_path: Path,
    plan: dict,
    *,
    plan_artifact_path: Optional[Path] = None,
) -> None:
    """Save plan to workflow config."""
    with open(config_path) as f:
        config = json.load(f)
    config["plan"] = plan
    if plan_artifact_path is not None:
        config["plan_artifact_path"] = str(plan_artifact_path)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
