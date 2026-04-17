#!/usr/bin/env python3
"""
VERITAS — Full Workflow Runner

Four-phase agentic pipeline for biomedical hypothesis testing on imaging datasets.
Agents autonomously plan, execute, and interpret hypothesis tests.

Phases:
- Phase 1: Agents query dataset and create analysis plan (groups, structures, observations, metrics, test)
- Phase 2A: Code-driven segmentation request preparation + MCP execution
- Phase 2B: Statistical analysis with code generation
- Phase 3: Team interpretation and verdict

    Input: Hypothesis + Dataset Metadata → Output: Evidence-labeled verdict + audit trail
"""

import os
import sys
import subprocess
import json
import argparse
import tempfile
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from veritas.workflow.audit import (
        map_transient_reason_to_failure_code,
        update_workflow_audit,
    )
except Exception:
    from audit import (  # type: ignore
        map_transient_reason_to_failure_code,
        update_workflow_audit,
    )

GLOBAL_CONTEXT_ENV = "VERITAS_CONTEXT_LENGTH"
GLOBAL_MAX_OUTPUT_ENV = "VERITAS_MAX_OUTPUT_TOKENS"


def _read_global_context_override() -> Optional[int]:
    """Read optional global context-length override from environment."""
    value = os.environ.get(GLOBAL_CONTEXT_ENV)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


@dataclass
class WorkflowConfig:
    """Configuration for the VERITAS workflow.

    Agents discover dataset structure via the SAT API and create analysis plans
    that generalize across hypothesis types (group difference, correlation, survival).
    """

    # Global settings
    llm_provider: str = "ollama"
    context_length: int = 16384
    max_output_tokens: int = 16384
    prompt_verbosity: str = "standard"
    temperature: float = 0.2

    # Per-role temperature overrides (optional)
    # Keys: "pi", "imaging_discussion", "imaging_execution",
    #        "statistician_discussion", "statistician_coding", "critic"
    temperatures: Optional[dict] = None

    # Global top_p (nucleus sampling). If None, provider default is used.
    top_p: Optional[float] = None
    # Per-role top_p overrides (optional, same keys as temperatures)
    top_ps: Optional[dict] = None

    # Models (nested dict gets flattened on load)
    models: Optional[dict] = None

    # Legacy flat model fields (for backwards compatibility)
    pi_model: Optional[str] = None
    imaging_discussion_model: Optional[str] = None
    imaging_execution_model: Optional[str] = None
    statistician_discussion_model: Optional[str] = None
    statistician_coding_model: Optional[str] = None
    critic_model: Optional[str] = None

    # Research parameters
    hypothesis: str = ""  # Research question in natural language
    dataset_path: str = ""  # Path to dataset (required)

    # Dataset context (user-provided)
    dataset: Optional[dict] = None
    # Example:
    # {
    #   "name": "acdc",
    #   "domain": "cardiac_mri",
    #   "modality": "Cine MRI (short-axis stack)",
    #   "available_groups": ["DCM", "HCM", "MINF", "NOR", "RV"],
    #   "available_observations": ["ED", "ES"],
    #   "patient_metadata_fields": ["height", "weight", "ed_frame", "es_frame"],
    #   "domain_notes": "Free-form text with formulas, temporal info, conventions, etc."
    # }

    # SAT cache database (optional, for centralized caching)
    sat_cache_database: Optional[str] = None

    # Ablation: use ground truth segmentations instead of SAT predictions
    use_ground_truth: bool = False

    # Ablation: enable/disable critic in Phase 2A and 2B
    enable_critic: bool = True
    # Optional ablations: enable critic in discussion phases
    enable_critic_phase1: bool = False
    enable_critic_phase3: bool = False

    # Ablation: override SAT segmentation model variant (default: agent chooses, typically "nano")
    sat_model_variant: Optional[str] = None

    # Phase control
    run_phase1: bool = True
    run_phase2a: bool = True
    run_phase2b: bool = True
    run_phase3: bool = True

    # Output settings
    output_path: str = "outputs/workflow"

    # Experiment metadata
    experiment_name: str = "veritas_experiment"
    description: str = "Dataset-driven biomarker discovery"
    analysis_constraints: Optional[dict] = None
    max_transient_retries_per_phase: int = 1

    @classmethod
    def from_json(cls, path: Path) -> 'WorkflowConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Flatten nested models dict for backwards compatibility
        if "models" in data and isinstance(data["models"], dict):
            models = data.pop("models")
            data["pi_model"] = models.get("pi")
            data["imaging_discussion_model"] = models.get("imaging_discussion")
            data["imaging_execution_model"] = models.get("imaging_execution")
            data["statistician_discussion_model"] = models.get("statistician_discussion")
            data["statistician_coding_model"] = models.get("statistician_coding")
            data["critic_model"] = models.get("critic")

        context_override = _read_global_context_override()
        if context_override is not None:
            data["context_length"] = context_override

        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def apply_to_environment(self) -> None:
        """Apply configuration to environment variables."""
        context_override = _read_global_context_override()
        if context_override is not None:
            self.context_length = context_override

        os.environ["LLM_PROVIDER"] = self.llm_provider
        os.environ[GLOBAL_CONTEXT_ENV] = str(self.context_length)
        os.environ[GLOBAL_MAX_OUTPUT_ENV] = str(self.max_output_tokens)
        os.environ["OLLAMA_CONTEXT_LENGTH"] = str(self.context_length)
        os.environ["OLLAMA_MODEL"] = self.statistician_coding_model
        if "OLLAMA_KEEP_ALIVE" not in os.environ:
            os.environ["OLLAMA_KEEP_ALIVE"] = "0"
        os.environ["WORKFLOW_OUTPUT_PATH"] = self.output_path
        os.environ["WORKFLOW_PROMPT_VERBOSITY"] = self.prompt_verbosity
        os.environ["WORKFLOW_ENABLE_CRITIC"] = "1" if self.enable_critic else "0"
        os.environ["WORKFLOW_ENABLE_CRITIC_PHASE1"] = "1" if self.enable_critic_phase1 else "0"
        os.environ["WORKFLOW_ENABLE_CRITIC_PHASE3"] = "1" if self.enable_critic_phase3 else "0"

        # Pass dataset info to phases
        if self.dataset_path:
            os.environ["DATASET_PATH"] = self.dataset_path

        if self.dataset:
            os.environ["DATASET_METADATA"] = json.dumps(self.dataset)


def run_phase(
    phase_num: str,
    phase_name: str,
    script_name: str,
    config: WorkflowConfig,
    config_file: Path
) -> int:
    """Run a single phase of the workflow."""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {phase_name}")
    print("=" * 80 + "\n")
    print(f"Configuration:")
    print(f"  • Context length: {config.context_length} tokens")
    print(f"  • Prompt verbosity: {config.prompt_verbosity}")
    print(f"  • Dataset: {config.dataset_path}")

    if phase_num in ["1", "3"]:
        print(f"  • PI model: {config.pi_model}")
        print(f"  • Imaging (discussion): {config.imaging_discussion_model}")
        print(f"  • Statistician (discussion): {config.statistician_discussion_model}")
        if phase_num == "1":
            print(f"  • Critic enabled: {config.enable_critic_phase1}")
        else:
            print(f"  • Critic enabled: {config.enable_critic_phase3}")
    elif phase_num == "2A":
        print(f"  • Imaging (execution): {config.imaging_execution_model}")
        print(f"  • Critic: {config.critic_model}")
    elif phase_num == "2B":
        print(f"  • Statistician (coding): {config.statistician_coding_model}")
        print(f"  • Critic: {config.critic_model}")

    print(f"  • Temperature: {config.temperature}")
    if config.temperatures:
        for role, temp in config.temperatures.items():
            print(f"    - {role}: {temp}")
    print()

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = Path(__file__).parent / "phases" / script_name

    env = os.environ.copy()
    env["WORKFLOW_CONFIG_FILE"] = str(config_file)

    output_base = Path(config.output_path)
    if not output_base.is_absolute():
        output_base = repo_root / output_base
    output_base.mkdir(parents=True, exist_ok=True)
    log_file = output_base / f"phase{phase_num}.log"

    _default_timeout = 1800 if str(phase_num).upper() == "2B" else 1200
    phase_timeout = int(os.environ.get("VERITAS_PHASE_TIMEOUT", str(_default_timeout)))
    max_retries = max(0, int(getattr(config, "max_transient_retries_per_phase", 0)))
    attempt = 0
    while True:
        with open(log_file, "w") as log:
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=repo_root,
                    env=env,
                    stderr=log,
                    timeout=phase_timeout,
                )
            except subprocess.TimeoutExpired:
                if attempt < max_retries:
                    attempt += 1
                    print(
                        f"\n⚠ Phase {phase_num} timed out after {phase_timeout // 60} minutes "
                        f"(retry {attempt}/{max_retries})"
                    )
                    continue
                print(f"\n❌ Phase {phase_num} timed out after {phase_timeout // 60} minutes")
                print(f"   Stderr log: {log_file}")
                _record_runner_failure(
                    config_file=config_file,
                    phase_num=phase_num,
                    failure_code="INFRA_TIMEOUT",
                    failure_message=f"Phase {phase_num} timed out after {phase_timeout // 60} minutes",
                    details={"timeout_seconds": phase_timeout, "log_file": str(log_file)},
                )
                return 1

        if result.returncode == 0:
            break

        retry_reason = _classify_transient_phase_failure(log_file)
        if retry_reason and attempt < max_retries:
            attempt += 1
            print(
                f"\n⚠ Phase {phase_num} failed with transient error ({retry_reason}) "
                f"(retry {attempt}/{max_retries})"
            )
            continue

        print(f"\n❌ Phase {phase_num} failed with exit code {result.returncode}")
        print(f"   Stderr log: {log_file}")
        failure_code = (
            map_transient_reason_to_failure_code(retry_reason)
            if retry_reason
            else "INFRA_UNKNOWN"
        )
        log_tail = _read_log_tail(log_file)
        failure_message = (
            f"Phase {phase_num} exited with code {result.returncode}"
            + (f" ({retry_reason})" if retry_reason else "")
        )
        if log_tail:
            failure_message = f"{failure_message} | log_tail: {log_tail}"
        _record_runner_failure(
            config_file=config_file,
            phase_num=phase_num,
            failure_code=failure_code,
            failure_message=failure_message,
            details={
                "exit_code": result.returncode,
                "retry_reason": retry_reason,
                "log_file": str(log_file),
                "attempt": attempt,
                "max_retries": max_retries,
            },
        )
        return result.returncode

    print(f"\n✓ Phase {phase_num} completed successfully")
    return 0


def _classify_transient_phase_failure(log_file: Path) -> Optional[str]:
    if not log_file.exists():
        return None

    text = log_file.read_text(errors="ignore")
    if not text.strip():
        return "empty_log"

    tail = text[-12000:]
    transient_patterns = [
        ("ollama_stream", r"ollama\._types\.ResponseError|error parsing tool call"),
        ("connection_error", r"Connection refused|Failed to establish a new connection|Connection reset"),
        ("transport_timeout", r"Read timed out|timed out"),
        ("import_error", r"ModuleNotFoundError|ImportError"),
        ("runtime_library_error", r"No module named"),
    ]
    for reason, pattern in transient_patterns:
        if re.search(pattern, tail, re.IGNORECASE):
            return reason

    return None


def _phase_stage_from_num(phase_num: str) -> str:
    mapping = {
        "1": "phase1",
        "2A": "phase2a",
        "2B": "phase2b",
        "3": "phase3",
    }
    return mapping.get(str(phase_num).upper(), "infra")


def _read_log_tail(log_file: Path, max_chars: int = 8000) -> str:
    if not log_file.exists():
        return ""
    try:
        return log_file.read_text(errors="ignore")[-max_chars:]
    except Exception:
        return ""


def _record_runner_failure(
    *,
    config_file: Path,
    phase_num: str,
    failure_code: str,
    failure_message: str,
    details: Optional[dict] = None,
) -> None:
    """Persist fallback audit metadata when a phase exits before self-reporting."""
    stage = _phase_stage_from_num(phase_num)
    update_workflow_audit(
        config_file=config_file,
        phase=stage,
        status="failed",
        failure_code=failure_code,
        failure_message=failure_message[:2000],
        details=details or {},
    )


def _phase1_marked_untestable(config_file: Path) -> tuple[bool, dict]:
    """Check whether Phase 1 plan marked the hypothesis as untestable."""
    try:
        with open(config_file) as f:
            data = json.load(f)
    except Exception:
        return False, {}

    plan = data.get("plan", {})
    feasibility = plan.get("feasibility") if isinstance(plan, dict) else None
    if not isinstance(feasibility, dict):
        return False, {}

    status = str(feasibility.get("status") or "").strip().upper()
    return status == "UNTESTABLE", feasibility


def print_configuration(config: WorkflowConfig) -> None:
    """Pretty-print configuration."""
    print("\n" + "=" * 80)
    print("VERITAS - WORKFLOW CONFIGURATION")
    print("=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Description: {config.description}")
    print()
    print("Dataset:")
    print(f"  • Path: {config.dataset_path}")
    if config.dataset:
        if "name" in config.dataset:
            print(f"  • Name: {config.dataset['name']}")
        if "domain" in config.dataset:
            print(f"  • Domain: {config.dataset['domain']}")
        if "modality" in config.dataset:
            print(f"  • Modality: {config.dataset['modality']}")
        if "available_groups" in config.dataset:
            groups = ", ".join(config.dataset["available_groups"])
            print(f"  • Available groups: {groups}")
        if "available_observations" in config.dataset:
            obs = ", ".join(config.dataset["available_observations"])
            print(f"  • Available observations: {obs}")
    else:
        print(f"  • Metadata: Not provided (agents will discover dataset structure)")
    print()
    print("Environment:")
    print(f"  • Provider: {config.llm_provider}")
    print(f"  • Context Length: {config.context_length} tokens")
    print(f"  • Prompt Verbosity: {config.prompt_verbosity}")
    print()
    print("Models:")
    print(f"  • PI: {config.pi_model}")
    print(f"  • Imaging (discussion, Phase 1&3): {config.imaging_discussion_model}")
    print(f"  • Imaging (execution, Phase 2A): {config.imaging_execution_model}")
    print(f"  • Statistician (discussion, Phase 1&3): {config.statistician_discussion_model}")
    print(f"  • Statistician (coding, Phase 2B): {config.statistician_coding_model}")
    print(f"  • Critic model: {config.critic_model}")
    print(f"  • Critic enabled (Phase 1): {config.enable_critic_phase1}")
    print(f"  • Critic enabled (Phase 2A/2B): {config.enable_critic}")
    print(f"  • Critic enabled (Phase 3): {config.enable_critic_phase3}")
    print()
    print("Research:")
    print(f"  • Hypothesis: {config.hypothesis}")
    print()
    print("Sampling:")
    print(f"  • Temperature: {config.temperature}")
    if config.temperatures:
        for role, temp in config.temperatures.items():
            print(f"    - {role}: {temp}")
    if config.top_p is not None:
        print(f"  • Top-p: {config.top_p}")
    if config.top_ps:
        for role, tp in config.top_ps.items():
            print(f"    - {role}: {tp}")
    print(f"  • Max transient retries/phase: {config.max_transient_retries_per_phase}")
    print()
    print("Output:")
    print(f"  • Path: {config.output_path}")
    print()
    print("Phases:")
    print(f"  • Phase 1 (Hypothesis): {'✓' if config.run_phase1 else '✗'}")
    print(f"  • Phase 2A (Imaging): {'✓' if config.run_phase2a else '✗'}")
    print(f"  • Phase 2B (Statistics): {'✓' if config.run_phase2b else '✗'}")
    print(f"  • Phase 3 (Interpretation): {'✓' if config.run_phase3 else '✗'}")
    print("=" * 80)


def main():
    """Run complete VERITAS workflow."""
    parser = argparse.ArgumentParser(
        description="Run VERITAS 4-phase workflow (dataset-driven hypothesis testing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python workflow_runner.py

  # Use specific configuration file
  python workflow_runner.py -c configs/experiment_default.json

  # Specify dataset path directly
  python workflow_runner.py --dataset /path/to/dataset_root
        """
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset root directory (overrides config file)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive prompts (for batch execution)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute() and not config_path.exists():
            script_relative = script_dir / config_path
            if script_relative.exists():
                config_path = script_relative

        if not config_path.exists():
            print(f"\n❌ Configuration file not found: {config_path}")
            return 1

        print(f"\n📋 Loading configuration from {config_path}")
        config = WorkflowConfig.from_json(config_path)
    else:
        # Try default locations
        config_path = script_dir / "workflow_config.json"
        default_config = script_dir / "configs" / "experiment_default.json"

        if config_path.exists():
            print(f"\n📋 Loading configuration from {config_path}")
            config = WorkflowConfig.from_json(config_path)
        elif default_config.exists():
            print(f"\n📋 Loading default configuration from {default_config}")
            config = WorkflowConfig.from_json(default_config)
        else:
            config = WorkflowConfig()
            print("\n📋 Using hardcoded defaults")

    # Override dataset path if provided via CLI
    if args.dataset:
        config.dataset_path = args.dataset
        print(f"📂 Dataset path overridden: {config.dataset_path}")

    # Validate dataset path
    if not config.dataset_path:
        print("\n❌ Error: dataset_path must be specified in config or via --dataset flag")
        return 1

    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        print(f"\n❌ Error: Dataset path does not exist: {dataset_path}")
        return 1

    # Apply configuration to environment
    config.apply_to_environment()

    # Print configuration
    print_configuration(config)

    # Print header
    print("\n" + "=" * 80)
    print("VERITAS: AGENTIC HYPOTHESIS TESTING FRAMEWORK")
    print("=" * 80)
    print()
    print("Multi-agent workflow for image-derived biomedical hypothesis testing:")
    print("  • Phase 1: Team queries dataset and creates analysis plan")
    print("  • Phase 2A: Imaging specialist prepares segmentation request (code-driven)")
    print("  • Phase 2B: Statistician performs analysis and tests hypothesis (code-driven)")
    print("  • Phase 3: Team interprets findings and reaches verdict")
    print()
    print("🔬 Input: Hypothesis + Dataset Metadata")
    print("🎯 Output: Statistically validated verdict")
    print()

    if not args.non_interactive:
        input("Press Enter to begin the workflow...")
    else:
        print("Running in non-interactive mode...\n")

    # Determine which phases to run
    phases = []
    if config.run_phase1:
        phases.append(("1", "Hypothesis Planning", "phase1_hypothesis.py"))
    if config.run_phase2a:
        phases.append(("2A", "Segmentation Request (Code-Driven)", "phase2a_imaging.py"))
    if config.run_phase2b:
        phases.append(("2B", "Statistical Analysis (Code-Driven)", "phase2b_statistics.py"))
    if config.run_phase3:
        phases.append(("3", "Interpretation & Verdict", "phase3_interpretation.py"))

    if not phases:
        print("❌ No phases enabled in configuration")
        return 1

    # Write persistent workflow config for all phases to share
    output_base = Path(config.output_path).absolute()
    output_base.mkdir(parents=True, exist_ok=True)
    config_path = output_base / "workflow_config.json"
    config.to_json(config_path)

    terminated_untestable = False

    # Run selected phases
    for phase_num, phase_name, script_name in phases:
        exit_code = run_phase(phase_num, phase_name, script_name, config, config_path)
        if exit_code != 0:
            print(f"\n❌ Workflow aborted at Phase {phase_num}")
            return exit_code

        if phase_num == "1":
            is_untestable, feasibility = _phase1_marked_untestable(config_path)
            if is_untestable:
                print("\n⚠ Phase 1 determined the hypothesis is UNTESTABLE.")
                print("  Skipping Phases 2A, 2B, and 3.")
                print(f"  • Invalid subtype: {feasibility.get('invalid_subtype', 'UNTESTABLE_OTHER')}")
                if feasibility.get("reason"):
                    print(f"  • Reason: {feasibility.get('reason')}")
                terminated_untestable = True
                break

    # Workflow finished (either all phases completed, or early termination on untestable)
    print("\n" + "=" * 80)
    if terminated_untestable:
        print("✓ WORKFLOW COMPLETED (UNTESTABLE TERMINATION AFTER PHASE 1)")
    else:
        print("✓ ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")

    # Save configuration to outputs for reproducibility (overwrite persistent file)
    final_config = asdict(config)
    if config_path.exists():
        try:
            with open(config_path) as f:
                persisted = json.load(f)
            for key in ("plan", "verdict", "workflow_audit"):
                if key in persisted:
                    final_config[key] = persisted[key]
            for key, value in persisted.items():
                if key not in final_config:
                    final_config[key] = value
        except Exception:
            pass
    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"✓ Configuration saved to {config_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
