#!/usr/bin/env python3
"""Phase 3: Interpretation

Team interprets findings from Phase 2 and reaches a final verdict on the hypothesis.
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
sys.path.insert(0, str(repo_root / "experiments"))

from veritas import run_meeting
from veritas.prompts import (
    PRINCIPAL_INVESTIGATOR,
    PHASE_AWARE_CRITIC,
    MEDICAL_IMAGING_SPECIALIST_INTERPRETATION,
    ML_STATISTICIAN_DISCUSSION,
    create_agent_with_model,
)
from veritas.utils import archive_meeting_artifacts, build_correction_summary_sources, load_summaries
from veritas.verbosity import get_prompt_verbosity_config
from veritas.workflow.audit import (
    append_correction_report,
    classify_failure_code,
    update_workflow_audit,
    write_validation_report,
)
from evaluation import compute_evidence_label


@dataclass
class PhaseConfig:
    """Configuration for phase execution."""
    output_path: str = "outputs/workflow"
    pi_model: str = "qwen3:8b"
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
    dataset_metadata: Optional[dict] = None
    critic_model: str = "qwen3:8b"
    enable_critic_phase3: bool = False


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
            dataset_metadata=data.get("dataset"),
            critic_model=models.get("critic", data.get("critic_model", PhaseConfig.critic_model)),
            enable_critic_phase3=_parse_bool(
                data.get("enable_critic_phase3"),
                default=PhaseConfig.enable_critic_phase3,
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
        dataset_metadata=dataset_metadata,
        critic_model=os.environ.get("CRITIC_MODEL", PhaseConfig.critic_model),
        enable_critic_phase3=_parse_bool(
            os.environ.get("WORKFLOW_ENABLE_CRITIC_PHASE3"),
            default=PhaseConfig.enable_critic_phase3,
        ),
    )


def main():
    """Run Phase 3: Interpretation."""
    print("\n" + "=" * 70)
    print("PHASE 3: INTERPRETATION")
    print("=" * 70 + "\n")

    config = load_config()

    if not config.hypothesis:
        raise ValueError("No hypothesis provided in config")

    # Resolve output path
    output_path = Path(config.output_path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    save_dir = (output_path / "phase3_interpretation").absolute()
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = os.environ.get("WORKFLOW_CONFIG_FILE")
    if not config_path:
        raise RuntimeError("WORKFLOW_CONFIG_FILE not set")
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise RuntimeError(f"WORKFLOW_CONFIG_FILE not found: {config_path_obj}")

    update_workflow_audit(
        config_file=config_path_obj,
        phase="phase3",
        status="running",
        details={"script": "phase3_interpretation.py"},
    )

    correction_attempted = False
    correction_success = None
    verdict = None
    try:
        # Create agents
        pi = create_agent_with_model(PRINCIPAL_INVESTIGATOR, config.pi_model, config.pi_temperature, config.pi_top_p)
        imaging_specialist = create_agent_with_model(
            MEDICAL_IMAGING_SPECIALIST_INTERPRETATION,
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

        # Load workflow config for hypothesis metadata
        config_data = {}
        with open(config_path_obj) as f:
            config_data = json.load(f)

        hypothesis_id = config_data.get("experiment_name")
        hypothesis_def, dataset_registry = load_hypothesis_definition(
            hypothesis_id,
            config.hypothesis
        )
        phase2b_results = load_phase2b_results(output_path)
        evidence = compute_evidence_label(
            phase2b_results,
            hypothesis_def or {},
            dataset_registry,
            output_path
        )

        # Load results snapshot — evidence label excluded (agents reason from raw stats)
        results_snapshot = load_results_snapshot(output_path)

        segmentation_note = "Segmentation quality metrics (Dice scores) are NOT available since we have no ground truth — assume segmentations are adequate for analysis."

        # Build agenda - clean problem statement
        agenda = f"""**Hypothesis:** {config.hypothesis}

**Results from Previous Phases:**
{results_snapshot}

**Task:** Interpret the results and reach a verdict on whether the hypothesis is supported

**Discussion Focus:**
This is an INTERPRETATION phase - focus on TEXT-BASED discussion of existing results.
- Review what was found in Phase 2A (segmentation) and Phase 2B (statistics)
- Discuss implications, limitations, and confidence in the findings
- Reach a final verdict: YES (supported), NO (rejected), or INCONCLUSIVE
- Interpret ONLY the hypothesis test that was run; do NOT introduce or claim group differences or additional analyses unless they were explicitly performed in Phase 2B

**When Code is Appropriate (RARE):**
Code should ONLY be used if you need to:
- Verify calculations from Phase 2B results
- Compute additional metrics to clarify the verdict (e.g., confidence intervals)
- Check if results make statistical sense

Code is NOT needed for:
- Re-running Phase 2B analysis (already done)
- Re-running power analysis (Phase 1 already computed this — refer to Phase 1 summary)
- General discussion of statistical concepts

**Discussion Points:**
1. **Imaging**: Are there technical limitations that could affect interpretation? Note: {segmentation_note}

2. **Statistics**: What do the statistical results show?
   - Is p < 0.05? Is the effect in the expected direction?
   - Refer to the Phase 1 power estimate (in Phase 1 summary): Phase 1 computed a priori power at d=0.5 (medium effect). If Phase 1 found adequate power (>= 0.80), the study was designed to detect meaningful effects
   - Do NOT compute post-hoc power from the observed effect size — this is a known statistical fallacy that always yields low power when results are non-significant
   - Missing covariate adjustments or unavailable data are LIMITATIONS to note, not reasons for INCONCLUSIVE

3. **Verdict**: Based on all evidence, is the hypothesis supported, rejected, or inconclusive?

**Verdict Guidelines:**
- YES: Statistical significance (p < 0.05) AND effect in the direction claimed by the hypothesis. Always verify direction from `group_statistics` means (NOT effect_size sign — non-parametric effect sizes like rank_biserial can have arbitrary sign): if hypothesis says "A > B", confirm mean_A > mean_B. For survival analyses: "Group A survives LONGER" means mean_survival_days(A) > mean_survival_days(B) — higher survival_days = better outcome. A hazard ratio HR < 1 for group A means group A has LOWER hazard = LONGER survival. If the hypothesis says "Group A → SHORTER survival" but mean_survival_days(A) > mean_survival_days(B), the direction is OPPOSITE → verdict is NO, not YES.
- NO: Not statistically significant, OR effect direction is opposite to the hypothesis claim (even if significant). A non-significant result in an adequately powered study (Phase 1 power >= 0.80) is evidence AGAINST the hypothesis — verdict NO, not INCONCLUSIVE.
- INCONCLUSIVE: Use ONLY when BOTH conditions are met: (1) Phase 1 found the study was underpowered (power < 0.80) AND (2) the result is not statistically significant (p >= 0.05). This applies regardless of observed direction — do NOT upgrade to NO based on opposite direction alone when the study is underpowered.

Important distinctions:
- A significant result (p < 0.05) in an underpowered study is still YES — low power does not invalidate a detected effect
- Missing covariates, lack of Dice scores, or unavailable data are LIMITATIONS to note in reasoning, NOT reasons for INCONCLUSIVE
- UNDERPOWERED refers to a priori power from Phase 1, NOT post-hoc power from observed effect size
- INVALID is for fundamental methodological failures (data corruption, NaN results, completely wrong analysis type) — not for assumption violations, sample size mismatches, or unbalanced groups. A parametric test used when assumptions are mildly violated is a limitation, not INVALID
- Exception for INVALID: if evidence is INVALID (p=1.0 exactly or other impossible value) AND the group_statistics means CLEARLY violate the hypothesis direction (e.g., hypothesis says A < B but mean_A >> mean_B), verdict is NO. This exception does NOT apply to UNDERPOWERED evidence — underpowered studies remain INCONCLUSIVE even with opposite direction.
- Large effect sizes (Cohen's d > 2) are biologically plausible when comparing groups with fundamentally different pathophysiology (e.g., cardiomyopathy vs. normal heart, high-grade vs. low-grade tumors). Do NOT mark INCONCLUSIVE solely due to a large effect size. Instead, verify internal consistency: p-value, group_statistics means, and sample sizes should all be mutually consistent. If they are, a large d is a real finding.
"""

        # Summary instructions for PI
        summary_instructions = """
**CRITICAL - Required Output Format:**

Your summary MUST end with this JSON block as the FINAL element (nothing after it):

```json
{
  "verdict": "<YES|NO|INCONCLUSIVE>",
  "evidence_label": "<SUPPORTED|REFUTED|UNDERPOWERED|INVALID>",
  "p_value": <number from Phase 2B>,
  "effect_size": <number from Phase 2B>,
  "test_used": "<test name>",
  "sample_sizes": {"group1": N, "group2": N, ...},
  "confidence": "<high|medium|low>",
  "reasoning": "<one sentence conclusion>"
}
```

Field descriptions:
- `verdict`: YES (hypothesis supported), NO (hypothesis rejected), or INCONCLUSIVE
- `evidence_label`: Your assessment: SUPPORTED/REFUTED/UNDERPOWERED/INVALID based on your interpretation
- `p_value`: Statistical significance (from Phase 2B)
- `effect_size`: Magnitude of difference (from Phase 2B)
- `test_used`: Statistical test name
- `sample_sizes`: Dictionary mapping group names to sample sizes
- `confidence`: high/medium/low based on statistical power and quality
- `reasoning`: One-sentence summary of why this verdict was reached

Use the exact Phase 2B values for `p_value`, `effect_size`, `test_used`, and `sample_sizes` (do NOT invent).
If Phase 2B results are missing or incomplete, verdict MUST be "INCONCLUSIVE".

The JSON must be valid. Do NOT write any text after the JSON block.
"""

        print(f"Running interpretation meeting with:")
        print(f"  • PI: {config.pi_model}")
        print(f"  • Imaging Specialist: {config.imaging_discussion_model}")
        print(f"  • Statistician: {config.statistician_discussion_model}")
        print(f"  • Critic enabled: {config.enable_critic_phase3}")
        print(f"  • Temperature: {config.temperature}")
        print(f"  • Verbosity: {config.prompt_verbosity}")
        print()

        # Load summaries from previous phases
        phase1_json = (output_path / "phase1_hypothesis_formulation" / "discussion.json").absolute()
        phase2a_json = (output_path / "phase2a_imaging_analysis" / "discussion.json").absolute()
        phase2b_json = (output_path / "phase2b_statistical_analysis" / "discussion.json").absolute()

        summaries = load_summaries([p for p in [phase1_json, phase2a_json, phase2b_json] if p.exists()])

        run_meeting(
            meeting_type="team",
            team_lead=pi,
            team_members=(imaging_specialist, statistician),
            agenda=agenda,
            summaries=summaries,
            save_dir=save_dir,
            critic=critic,
            num_rounds=1,
            temperature=config.temperature,
            top_p=config.top_p,
            prompt_verbosity=config.prompt_verbosity,
            enable_critic=config.enable_critic_phase3,
            workflow_instruction="Phase 3: Interpretation - Interpret results and reach verdict",
            summary_instructions=summary_instructions,
        )

        print(f"\n✓ Phase 3 complete. Results saved to {save_dir}")
        print(f"  • Discussion: {save_dir / 'discussion.json'}")
        print(f"  • Summary: {save_dir / 'summary.md'}")

        summary_text = load_summary(save_dir)
        verdict = extract_verdict_block(summary_text)
        verdict_issues = validate_verdict_contract(verdict)

        if verdict_issues:
            correction_attempted = True
            print("\n  ⚠ Verdict JSON validation failed:")
            for issue in verdict_issues:
                print(f"    - {issue}")
            print("  → Requesting one correction pass from PI...")
            _request_verdict_correction(
                save_dir=save_dir,
                phase1_json=phase1_json,
                phase2a_json=phase2a_json,
                phase2b_json=phase2b_json,
                pi=pi,
                imaging_specialist=imaging_specialist,
                statistician=statistician,
                temperature=config.temperature,
                top_p=config.top_p,
                prompt_verbosity=config.prompt_verbosity,
                issues=verdict_issues,
                enable_critic=config.enable_critic_phase3,
                critic=critic,
                correction_index=1,
            )

            summary_text = load_summary(save_dir)
            verdict = extract_verdict_block(summary_text)
            remaining_issues = validate_verdict_contract(verdict)
            if remaining_issues:
                correction_success = False
                append_correction_report(
                    save_dir=save_dir,
                    phase="phase3",
                    attempted=True,
                    success=False,
                    reason="verdict_contract_still_invalid_after_correction",
                    error_before="; ".join(verdict_issues),
                    error_after="; ".join(remaining_issues),
                )
                raise ValueError(
                    "Verdict JSON invalid after correction. Issues: "
                    + "; ".join(remaining_issues)
                )

            correction_success = True
            append_correction_report(
                save_dir=save_dir,
                phase="phase3",
                attempted=True,
                success=True,
                reason="verdict_contract_correction_succeeded",
                error_before="; ".join(verdict_issues),
            )

        if not verdict:
            raise ValueError("Verdict JSON block not found in Phase 3 summary.")

        computed_label = evidence.get("evidence_label")
        if computed_label:
            verdict["computed_evidence_label"] = computed_label
            agent_label = verdict.get("evidence_label")
            if agent_label and str(agent_label).upper() != str(computed_label).upper():
                print(
                    f"  ⚠ Agent evidence label '{agent_label}' differs from "
                    f"computed label '{computed_label}'"
                )
        _check_verdict_evidence_alignment(verdict)
        update_config_verdict(config_path_obj, verdict)
        print(f"  • Verdict: {verdict.get('verdict', 'unknown')}")
        print(f"  • Evidence label: {verdict.get('evidence_label', 'unknown')}")
        print(f"  • Verdict saved to config: {config_path_obj}")

        write_validation_report(
            save_dir=save_dir,
            phase="phase3",
            status="passed",
            checks={
                "verdict_contract_valid": True,
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase3",
            status="passed",
            details={
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
    except Exception as e:
        failure_code = classify_failure_code("phase3", str(e))
        write_validation_report(
            save_dir=save_dir,
            phase="phase3",
            status="failed",
            checks={
                "verdict_contract_valid": False,
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
            errors=[str(e)],
            failure_code=failure_code,
        )
        update_workflow_audit(
            config_file=config_path_obj,
            phase="phase3",
            status="failed",
            failure_code=failure_code,
            failure_message=str(e),
            details={
                "correction_attempted": correction_attempted,
                "correction_success": correction_success,
            },
        )
        raise


def validate_verdict_contract(verdict: dict) -> list[str]:
    """Validate required Phase 3 verdict contract."""
    if not verdict or not isinstance(verdict, dict):
        return ["Missing verdict JSON object"]

    issues: list[str] = []
    required = [
        "verdict",
        "evidence_label",
        "p_value",
        "effect_size",
        "test_used",
        "sample_sizes",
        "confidence",
        "reasoning",
    ]
    missing = [key for key in required if key not in verdict]
    if missing:
        issues.append(f"Missing required keys: {', '.join(missing)}")

    verdict_value = str(verdict.get("verdict", "")).strip().upper()
    if verdict_value and verdict_value not in {"YES", "NO", "INCONCLUSIVE"}:
        issues.append("verdict must be one of YES, NO, INCONCLUSIVE")

    evidence_label = str(verdict.get("evidence_label", "")).strip().upper()
    if evidence_label and evidence_label not in {"SUPPORTED", "REFUTED", "UNDERPOWERED", "INVALID"}:
        issues.append("evidence_label must be one of SUPPORTED, REFUTED, UNDERPOWERED, INVALID")

    confidence = str(verdict.get("confidence", "")).strip().lower()
    if confidence and confidence not in {"high", "medium", "low"}:
        issues.append("confidence must be one of high, medium, low")

    if "reasoning" in verdict and not str(verdict.get("reasoning", "")).strip():
        issues.append("reasoning must be a non-empty string")

    if "test_used" in verdict and not str(verdict.get("test_used", "")).strip():
        issues.append("test_used must be a non-empty string")

    if "sample_sizes" in verdict and not isinstance(verdict.get("sample_sizes"), dict):
        issues.append("sample_sizes must be a dictionary")

    return issues


def _request_verdict_correction(
    correction_index: int,
    *,
    save_dir: Path,
    phase1_json: Path,
    phase2a_json: Path,
    phase2b_json: Path,
    pi,
    imaging_specialist,
    statistician,
    critic,
    temperature: float,
    top_p: Optional[float] = None,
    prompt_verbosity: str,
    issues: list[str],
    enable_critic: bool,
) -> None:
    correction_agenda = (
        "Regenerate the final Phase 3 summary with a valid final JSON block.\n"
        f"Validation issues: {'; '.join(issues)}\n\n"
        "Required:\n"
        "1) Keep scientific conclusions unchanged; fix contract only\n"
        "2) Final JSON includes: verdict, evidence_label, p_value, effect_size, test_used, sample_sizes, confidence, reasoning\n"
        "3) verdict in {YES, NO, INCONCLUSIVE}\n"
        "4) evidence_label in {SUPPORTED, REFUTED, UNDERPOWERED, INVALID}\n"
        "5) No text after final JSON block\n"
    )
    summary_sources = build_correction_summary_sources(
        prior_paths=[phase1_json, phase2a_json, phase2b_json],
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
        meeting_type="team",
        team_lead=pi,
        team_members=(imaging_specialist, statistician),
        agenda=correction_agenda,
        summaries=summaries,
        save_dir=save_dir,
        critic=critic,
        num_rounds=1,
        temperature=temperature,
        top_p=top_p,
        prompt_verbosity=prompt_verbosity,
        enable_critic=enable_critic,
        workflow_instruction="Phase 3: Verdict Contract Correction",
    )


def load_summary(save_dir: Path) -> str:
    """Load and return summary text."""
    summary_path = save_dir / "summary.md"
    if summary_path.exists():
        return summary_path.read_text()

    # Fall back to last message in discussion.json
    discussion_path = save_dir / "discussion.json"
    if not discussion_path.exists():
        raise FileNotFoundError(f"No summary or discussion found in {save_dir}")

    with open(discussion_path) as f:
        discussion = json.load(f)

    if not discussion:
        raise ValueError("Discussion is empty")

    return discussion[-1].get("message", "")


def load_results_snapshot(output_base: Path) -> str:
    """Load results snapshot from Phase 2A and 2B.

    Does NOT include the evaluator-computed evidence label — agents must
    reason from raw statistics to reach their own verdict.
    """
    lines = []

    # Phase 2A execution
    execution_path = output_base / "phase2a_imaging_analysis" / "segmentation_execution.json"
    if execution_path.exists():
        with open(execution_path) as f:
            exec_data = json.load(f)
        lines.append("**Phase 2A Segmentation:**")
        lines.append(f"- Success: {exec_data.get('success', False)}")
        lines.append(f"- Total identifiers: {exec_data.get('total_identifiers', 0)}")
        lines.append(f"- Structures: {', '.join(exec_data.get('structures', []))}")
        lines.append(f"- Cached: {exec_data.get('cached_count', 0)}, Processed: {exec_data.get('processed_count', 0)}")
        lines.append("")

    # Phase 2B results
    results_path = output_base / "phase2b_statistical_analysis" / "workspace" / "data" / "statistical_results.json"
    if results_path.exists():
        with open(results_path) as f:
            stats_data = json.load(f)
        lines.append("**Phase 2B Statistical Analysis:**")
        lines.append(f"- Test: {stats_data.get('test_performed', 'unknown')}")
        lines.append(f"- P-value: {stats_data.get('p_value', 'N/A')}")
        lines.append(f"- Effect size: {stats_data.get('effect_size', 'N/A')} ({stats_data.get('effect_size_type', 'unknown')})")
        lines.append(f"- Sample sizes: {stats_data.get('sample_sizes', {})}")
        if "group_statistics" in stats_data:
            lines.append(f"- Group statistics: {json.dumps(stats_data['group_statistics'], indent=2)}")
        lines.append("")

    return "\n".join(lines) if lines else "(No results found from Phase 2)"


def load_phase2b_results(output_base: Path) -> Optional[dict]:
    """Load statistical results from Phase 2B if available."""
    results_path = output_base / "phase2b_statistical_analysis" / "workspace" / "data" / "statistical_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def load_hypothesis_definition(hypothesis_id: Optional[str], hypothesis_text: str) -> tuple[Optional[dict], Optional[dict]]:
    """Load hypothesis definition and dataset registry from hypothesis bank."""
    bank_path = repo_root / "experiments" / "tiered_hypothesis_bank.json"
    if not bank_path.exists():
        return None, None
    with open(bank_path) as f:
        bank = json.load(f)
    dataset_registry = bank.get("dataset_registry")
    hypotheses = bank.get("hypotheses", [])
    if hypothesis_id:
        for hyp in hypotheses:
            if hyp.get("id") == hypothesis_id:
                return hyp, dataset_registry
    for hyp in hypotheses:
        if hyp.get("hypothesis") == hypothesis_text:
            return hyp, dataset_registry
    return None, dataset_registry


def _check_verdict_evidence_alignment(verdict: dict) -> None:
    """Log a warning if agent verdict doesn't match evidence label.

    Does NOT override the agent's verdict — this is advisory only.
    The agent is given the evidence label as context in the agenda and
    should reach the right conclusion on its own.
    """
    if not verdict:
        return
    evidence_label = verdict.get("evidence_label")
    if not evidence_label:
        return

    mapping = {
        "SUPPORTED": "YES",
        "REFUTED": "NO",
        "UNDERPOWERED": "INCONCLUSIVE",
        "INVALID": "INCONCLUSIVE",
    }
    expected_verdict = mapping.get(str(evidence_label).upper())
    if expected_verdict and verdict.get("verdict") != expected_verdict:
        print(
            f"  ⚠ Verdict-evidence mismatch: agent said {verdict.get('verdict')}, "
            f"evidence label suggests {expected_verdict} (from {evidence_label})"
        )
        verdict.setdefault("warnings", []).append("verdict_evidence_mismatch")


def extract_verdict_block(text: str) -> dict:
    """Extract verdict JSON from text.

    Required keys: verdict, confidence, reasoning
    Optional keys: p_value, effect_size, test_used, sample_sizes
    """
    required_keys = ["verdict", "confidence", "reasoning", "evidence_label"]
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]

    last_valid = None
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                verdict = json.loads(match.group(1))
                if "verdict" in verdict:
                    missing = [k for k in required_keys if k not in verdict]
                    if missing:
                        print(f"  ⚠ Verdict missing keys: {missing}")
                    last_valid = verdict
            except json.JSONDecodeError:
                continue

    if last_valid:
        return last_valid

    # Try any JSON object with "verdict" key
    for match in re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", text):
        try:
            verdict = json.loads(match.group(0))
            if "verdict" in verdict:
                return verdict
        except json.JSONDecodeError:
            continue

    return {}


def update_config_verdict(config_path: Path, verdict: dict) -> None:
    """Save verdict to workflow config."""
    with open(config_path) as f:
        config = json.load(f)
    config["verdict"] = verdict
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
