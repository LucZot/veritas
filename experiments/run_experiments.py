#!/usr/bin/env python3
"""
Batch experiment runner for hypothesis testing.

Runs multiple hypotheses through the workflow, supports repeated runs
for consistency testing, and generates evaluation reports.

Usage:
  # Run pilot (all 6 hypotheses, 1 run each)
  python run_experiments.py --pilot

  # Run specific hypotheses
  python run_experiments.py --hypotheses cardiac_01_dcm_lvef_lower cardiac_03_dcm_lvef_higher

  # Run with multiple repetitions for consistency testing
  python run_experiments.py --hypotheses cardiac_01_dcm_lvef_lower --n-runs 3

  # Run all hypotheses with 3 repetitions
  python run_experiments.py --n-runs 3

  # Run with a custom hypothesis bank
  python run_experiments.py --bank experiments/tiered_hypothesis_bank.json --n-runs 1

  # Run experiments and evaluate results
  python run_experiments.py --hypotheses cardiac_01_dcm_lvef_lower --n-runs 3 --evaluate
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from evaluation import extract_workflow_verdict, compute_metrics, print_summary, evaluate_batch

DATA_ROOT_PLACEHOLDER = "__DATA_ROOT__"
GLOBAL_CONTEXT_ENV = "VERITAS_CONTEXT_LENGTH"


def _resolve_data_root_in_registry(registry: dict) -> dict:
    """Replace __DATA_ROOT__ placeholders in dataset registry values."""
    data_root = os.environ.get("BIO_DATA_ROOT", "")
    for dataset_name, dataset_config in registry.items():
        for key, value in dataset_config.items():
            if isinstance(value, str) and DATA_ROOT_PLACEHOLDER in value:
                dataset_config[key] = value.replace(DATA_ROOT_PLACEHOLDER, data_root)
    return registry


def _extract_hypothesis_number(hypothesis_id: str) -> Optional[int]:
    """Extract numeric hypothesis index from IDs like cardiac_01_... or glioma_12_..."""
    match = re.search(r"_(\d+)(?:_|$)", str(hypothesis_id))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _parse_hypothesis_number_spec(spec: str) -> List[int]:
    """Parse comma-separated number/range specs (e.g., '1-8,10,12-14')."""
    values = set()
    if not spec:
        return []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return sorted(values)


def _resolve_hypothesis_filter_ids(
    hypotheses: List[Dict[str, Any]],
    requested_numbers: List[int],
) -> List[str]:
    """Resolve numeric selections to full hypothesis IDs."""
    requested = set(requested_numbers)
    selected_ids = []
    found_numbers = set()
    for hypothesis in hypotheses:
        hyp_number = _extract_hypothesis_number(hypothesis.get("id", ""))
        if hyp_number is not None and hyp_number in requested:
            selected_ids.append(hypothesis["id"])
            found_numbers.add(hyp_number)

    missing = sorted(requested - found_numbers)
    if missing:
        print(f"Warning: requested hypothesis numbers not found in current selection: {missing}")

    return selected_ids


def _parse_tier_filter_values(raw_values: Optional[List[str]]) -> Optional[set[str]]:
    """Parse tier filter values from CLI args."""
    if not raw_values:
        return None

    allowed = {"L0", "L1", "L2", "L3", "L4", "L5"}
    parsed: set[str] = set()
    for raw in raw_values:
        for token in str(raw).split(","):
            tier = token.strip().upper()
            if not tier:
                continue
            if tier not in allowed:
                raise ValueError(f"Invalid tier '{tier}'. Allowed tiers: {sorted(allowed)}")
            parsed.add(tier)

    return parsed or None


def _resolve_context_length_override(cli_value: Optional[int]) -> Optional[int]:
    """Resolve a single global context-length override (CLI > env)."""
    if cli_value is not None:
        return cli_value if cli_value > 0 else None

    env_value = os.environ.get(GLOBAL_CONTEXT_ENV)
    if not env_value:
        return None
    try:
        parsed = int(env_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _classify_run(run_dir: Path) -> str:
    """Classify a run directory as complete, agent_failure, interrupted, or missing.

    - complete      : Phase 3 discussion.json exists (successful workflow)
    - agent_failure : validation_report with non-INFRA failure code (valid data, skip)
    - interrupted   : run dir exists but killed by SLURM/infra (safe to re-run)
    - missing       : run dir doesn't exist
    """
    if not run_dir.exists():
        return "missing"

    # Complete: Phase 3 finished
    if (run_dir / "phase3_interpretation" / "discussion.json").exists():
        return "complete"
    if (run_dir / "phase3_interpretation" / "validation_report.json").exists():
        return "complete"

    # Complete: Phase 1 correctly identified hypothesis as untestable (L0 early stop)
    p1_report = run_dir / "phase1_hypothesis_formulation" / "validation_report.json"
    if p1_report.exists():
        try:
            with open(p1_report) as f:
                report = json.load(f)
            if report.get("status") == "passed" and report.get("checks", {}).get("untestable"):
                return "complete"
        except (json.JSONDecodeError, OSError):
            pass

    # Check for agent failure (non-INFRA failure codes in validation reports)
    for phase_name in [
        "phase1_hypothesis_formulation",
        "phase2a_imaging_analysis",
        "phase2b_statistical_analysis",
        "phase3_interpretation",
    ]:
        report_path = run_dir / phase_name / "validation_report.json"
        if not report_path.exists():
            continue
        try:
            with open(report_path) as f:
                report = json.load(f)
            status = report.get("status", "")
            failure_code = report.get("failure_code", "") or ""
            if status == "failed" and failure_code and not failure_code.startswith("INFRA_"):
                return "agent_failure"
        except (json.JSONDecodeError, OSError):
            continue

    # Also check workflow_config.json audit field
    wf_config = run_dir / "workflow_config.json"
    if wf_config.exists():
        try:
            with open(wf_config) as f:
                config = json.load(f)
            fc = config.get("workflow_audit", {}).get("failure_code", "") or ""
            if fc and not fc.startswith("INFRA_"):
                return "agent_failure"
        except (json.JSONDecodeError, OSError):
            pass

    # Run dir exists but no completion and no agent failure → infra interruption
    return "interrupted"


def create_experiment_config(
    hypothesis: Dict,
    base_config: Path,
    output_dir: Path,
    dataset_registry: Dict = None,
    use_ground_truth: bool = False,
    context_length_override: Optional[int] = None,
    enable_critic: bool = True,
    max_transient_retries: Optional[int] = None,
) -> Path:
    """
    Create experiment config from hypothesis and base config.

    Args:
        hypothesis: Hypothesis definition from bank
        base_config: Path to base config file
        output_dir: Directory for this experiment run
        dataset_registry: Dataset defaults to merge with hypothesis dataset config

    Returns:
        Path to created config file
    """
    with open(base_config) as f:
        config = json.load(f)

    # Resolve __DATA_ROOT__ in base config paths
    data_root = os.environ.get("BIO_DATA_ROOT", "")
    for key in ("dataset_path", "sat_cache_database"):
        if isinstance(config.get(key), str) and DATA_ROOT_PLACEHOLDER in config[key]:
            config[key] = config[key].replace(DATA_ROOT_PLACEHOLDER, data_root)

    # Update hypothesis
    config["hypothesis"] = hypothesis["hypothesis"]

    # Merge dataset config from registry and hypothesis overrides
    hypothesis_dataset = hypothesis.get("dataset", hypothesis.get("dataset_config", {}))
    dataset_name = hypothesis_dataset.get("name", hypothesis_dataset.get("dataset_name", "acdc"))

    # Merge from registry if available
    base_dataset_name = config.get("dataset", {}).get("name", "acdc")
    switching_dataset = dataset_name != base_dataset_name

    if dataset_registry and dataset_name in dataset_registry:
        registry_defaults = dataset_registry[dataset_name].copy()

        # Resolve dataset_path and sat_cache_database from registry
        if "dataset_path" in registry_defaults:
            config["dataset_path"] = registry_defaults.pop("dataset_path")
        if "sat_cache_database" in registry_defaults:
            config["sat_cache_database"] = registry_defaults.pop("sat_cache_database")

        if switching_dataset:
            # Different dataset: registry replaces base config dataset fields
            config["dataset"] = registry_defaults
        else:
            # Same dataset: merge (base config takes precedence)
            if "dataset" not in config:
                config["dataset"] = {}
            for key, value in registry_defaults.items():
                if key not in config["dataset"]:
                    config["dataset"][key] = value

    # Apply hypothesis-specific overrides (highest precedence)
    for key, value in hypothesis_dataset.items():
        # Map old dataset_config keys to new dataset keys
        if key == "dataset_name":
            config["dataset"]["name"] = value
        else:
            config["dataset"][key] = value

    # Ablation flags
    if use_ground_truth:
        config["use_ground_truth"] = True
    if not enable_critic:
        config["enable_critic"] = False

    if context_length_override is not None:
        config["context_length"] = int(context_length_override)
    if max_transient_retries is not None:
        config["max_transient_retries_per_phase"] = int(max(0, max_transient_retries))

    # Update output path and experiment name
    config["output_path"] = str(output_dir)
    config["experiment_name"] = hypothesis["id"]

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def run_single_workflow(
    config_path: Path,
    timeout: int = 3600,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single workflow with given config.

    Args:
        config_path: Path to experiment config
        timeout: Maximum runtime in seconds
        verbose: Print progress messages

    Returns:
        Dict with success status, duration, and any error info
    """
    repo_root = Path(__file__).parent.parent
    workflow_runner = repo_root / "src" / "veritas" / "workflow" / "runner.py"

    if not workflow_runner.exists():
        return {
            "success": False,
            "duration": 0,
            "error": f"Workflow runner not found: {workflow_runner}"
        }

    start_time = time.time()

    try:
        if verbose:
            # Run with output visible
            result = subprocess.run(
                ["python", str(workflow_runner), "-c", str(config_path), "--non-interactive"],
                timeout=timeout,
                cwd=repo_root
            )
        else:
            # Capture output
            result = subprocess.run(
                ["python", str(workflow_runner), "-c", str(config_path), "--non-interactive"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=repo_root
            )

        duration = time.time() - start_time

        # Check if workflow actually completed (even if returncode != 0)
        # Sometimes subprocess returns non-zero due to cleanup issues
        success = result.returncode == 0

        stdout = getattr(result, 'stdout', None) if hasattr(result, 'stdout') else None
        stderr = getattr(result, 'stderr', None) if hasattr(result, 'stderr') else None
        return {
            "success": success,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": (stdout or "")[-5000:] if isinstance(stdout, str) else None,
            "stderr": (stderr or "")[-2000:] if isinstance(stderr, str) else None
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration": timeout,
            "error": f"Timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e)
        }


def run_experiments(
    hypothesis_bank: List[Dict],
    base_config: Path,
    output_dir: Path,
    n_runs: int = 1,
    run_start_index: int = 0,
    hypotheses_filter: Optional[List[str]] = None,
    timeout: int = 3600,
    verbose: bool = True,
    dataset_registry: Dict = None,
    use_ground_truth: bool = False,
    eval_sesoi_override: Optional[Any] = None,
    context_length_override: Optional[int] = None,
    enable_critic: bool = True,
    max_transient_retries: int = 1,
) -> List[Dict]:
    """
    Run experiments for selected hypotheses.

    Args:
        hypothesis_bank: List of hypothesis definitions
        base_config: Path to base config file
        output_dir: Base output directory
        n_runs: Number of runs per hypothesis
        hypotheses_filter: Optional list of hypothesis IDs to run
        timeout: Per-run timeout in seconds
        verbose: Print progress
        dataset_registry: Optional dataset registry for config merging

    Returns:
        List of result dictionaries
    """
    results = []

    # Filter hypotheses if specified
    if hypotheses_filter:
        hypothesis_bank = [h for h in hypothesis_bank if h["id"] in hypotheses_filter]

    if not hypothesis_bank:
        print("No hypotheses to run!")
        return results

    total = len(hypothesis_bank) * n_runs
    current = 0

    print(f"\n{'='*60}")
    print(f"RUNNING {total} EXPERIMENTS")
    print(f"  Hypotheses: {len(hypothesis_bank)}")
    print(f"  Runs per hypothesis: {n_runs}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    skipped_complete = 0
    skipped_agent_fail = 0
    cleaned_interrupted = 0

    for hypothesis in hypothesis_bank:
        hyp_id = hypothesis["id"]

        for run_idx in range(run_start_index, run_start_index + n_runs):
            current += 1
            run_id = f"run_{run_idx:03d}"
            run_output_dir = output_dir / hyp_id / run_id

            # Classify existing run before doing anything
            run_status = _classify_run(run_output_dir)

            if run_status == "complete":
                skipped_complete += 1
                print(f"  [{current}/{total}] {hyp_id} ({run_id}) — SKIP (complete)")
                continue

            if run_status == "agent_failure":
                skipped_agent_fail += 1
                print(f"  [{current}/{total}] {hyp_id} ({run_id}) — SKIP (agent failure, valid data)")
                continue

            if run_status == "interrupted":
                # Clean up partially-written run before re-running
                print(f"\n[{current}/{total}] {hyp_id} ({run_id})")
                print(f"  Cleaning interrupted run dir: {run_output_dir}")
                shutil.rmtree(run_output_dir)
                cleaned_interrupted += 1
            else:
                print(f"\n[{current}/{total}] {hyp_id} ({run_id})")
            print(f"  Hypothesis: {hypothesis['hypothesis'][:60]}...")

            # Create config
            config_path = create_experiment_config(
                hypothesis, base_config, run_output_dir, dataset_registry,
                use_ground_truth=use_ground_truth,
                context_length_override=context_length_override,
                enable_critic=enable_critic,
                max_transient_retries=max_transient_retries,
            )

            # Run workflow
            print(f"  Starting workflow...")
            run_result = run_single_workflow(
                config_path, timeout=timeout, verbose=verbose
            )

            # Extract verdict
            if run_result["success"]:
                verdict = extract_workflow_verdict(
                    run_output_dir,
                    hyp_id,
                    run_id,
                    hypothesis.get("ground_truth"),
                    hypothesis=hypothesis,
                    dataset_registry=dataset_registry,
                    sesoi_override=eval_sesoi_override,
                )
                run_result["verdict"] = verdict.verdict
                run_result["is_correct"] = verdict.is_correct
                run_result["expected"] = verdict.expected_conclusion
                run_result["parse_method"] = verdict.parse_method
            else:
                run_result["verdict"] = "FAILED"
                run_result["is_correct"] = None

            results.append({
                "hypothesis_id": hyp_id,
                "run_id": run_id,
                "output_dir": str(run_output_dir),
                "timestamp": datetime.now().isoformat(),
                **run_result
            })

            # Print status
            status = "✓" if run_result["success"] else "✗"
            verdict_str = run_result.get("verdict", "N/A")
            expected_str = run_result.get("expected", "N/A")
            correct_str = "✓" if run_result.get("is_correct") else "✗" if run_result.get("is_correct") is False else "?"

            print(f"  {status} Completed in {run_result['duration']:.1f}s")
            print(f"  Verdict: {verdict_str} (expected: {expected_str}) [{correct_str}]")

    if skipped_complete or skipped_agent_fail or cleaned_interrupted:
        print(f"\n{'─'*60}")
        print(f"  Skipped {skipped_complete} complete, {skipped_agent_fail} agent failures")
        print(f"  Cleaned & re-ran {cleaned_interrupted} interrupted runs")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run hypothesis testing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run only PDGM hypotheses
  python run_experiments.py --dataset ucsf_pdgm

  # Run specific hypotheses with 3 repetitions
  python run_experiments.py --hypotheses cardiac_01_dcm_lvef_lower --n-runs 3

  # Run all hypotheses with custom config
  python run_experiments.py --config custom_config.json --n-runs 3

  # Run with a custom hypothesis bank filtered to one dataset
  python run_experiments.py --bank experiments/tiered_hypothesis_bank.json --dataset acdc
        """
    )
    parser.add_argument("--pilot", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=str,
                        help="Run only hypotheses for this dataset (e.g., acdc, ucsf_pdgm)")
    parser.add_argument("--tiers", nargs="+",
                        help="Filter by complexity tiers (e.g., L0 L1 or L2,L3)")
    hypothesis_select_group = parser.add_mutually_exclusive_group()
    hypothesis_select_group.add_argument("--hypotheses", nargs="+",
                                         help="Specific full hypothesis IDs to run")
    hypothesis_select_group.add_argument("--hypothesis-range", type=str,
                                         help="Numeric hypothesis range/list (e.g., 1-8 or 1-4,7,10)")
    hypothesis_select_group.add_argument("--hypothesis-numbers", nargs="+", type=int,
                                         help="Numeric hypothesis IDs (e.g., 1 2 7 8)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs per hypothesis (default: 3)")
    parser.add_argument("--run-start-index", type=int, default=0,
                        help="Starting run index, e.g. 5 to produce run_005..run_009 (default: 0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/experiments"),
                        help="Output directory for results")
    parser.add_argument("--bank", type=Path,
                        help="Path to hypothesis bank JSON (default: experiments/tiered_hypothesis_bank.json)")
    parser.add_argument("--config", type=Path,
                        help="Base config file (default: experiment_default.json)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-run timeout in seconds (default: 3600)")
    parser.add_argument("--context-length", type=int,
                        help="Global Ollama context length override for all runs/phases (e.g., 8192, 16384, 32768)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress workflow output")
    parser.add_argument(
        "--max-transient-retries",
        type=int,
        default=1,
        help="Retries per phase for transient infra failures (ollama stream/import/runtime), default: 1",
    )
    parser.add_argument("--use-ground-truth", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--disable-critic", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--tag", type=str,
                        help="Tag for this experiment run (appended to output dir name)")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--evaluate", dest="evaluate", action="store_true",
                            default=True, help="Run evaluation after experiments (default: on)")
    eval_group.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                            help="Skip evaluation step")
    sesoi_group = parser.add_mutually_exclusive_group()
    sesoi_group.add_argument(
        "--eval-sesoi-profile",
        choices=["strict", "standard", "loose"],
        help=argparse.SUPPRESS,
    )
    sesoi_group.add_argument(
        "--eval-sesoi-value",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--evaluation-output", type=Path,
                        help="Optional path to save evaluation results JSON")
    parser.add_argument("--exact-output-dir", type=Path,
                        help="Use this exact path as output directory (skips timestamp subdirectory creation). "
                             "Useful for Slurm job arrays where all tasks must write to the same directory.")
    args = parser.parse_args()

    # Determine paths
    experiments_dir = Path(__file__).parent
    repo_root = experiments_dir.parent

    # Load hypothesis bank (default remains experiments/hypothesis_bank.json)
    if args.bank:
        bank_path = args.bank
        # Support convenient relative paths from repo root and experiments/ dir.
        if not bank_path.is_absolute() and not bank_path.exists():
            experiments_relative = experiments_dir / bank_path
            if experiments_relative.exists():
                bank_path = experiments_relative
    else:
        bank_path = experiments_dir / "tiered_hypothesis_bank.json"

    if not bank_path.exists():
        print(f"Hypothesis bank not found: {bank_path}")
        return 1

    with open(bank_path) as f:
        bank = json.load(f)

    hypothesis_bank = bank["hypotheses"]
    dataset_registry = _resolve_data_root_in_registry(bank.get("dataset_registry", {}))

    # Filter by dataset if specified
    if args.dataset:
        dataset_filter = args.dataset.lower()
        hypothesis_bank = [
            h for h in hypothesis_bank
            if h.get("dataset", {}).get("name", "").lower() == dataset_filter
        ]
        if not hypothesis_bank:
            print(f"No hypotheses found for dataset '{args.dataset}'")
            available = set(
                h.get("dataset", {}).get("name", "unknown")
                for h in bank["hypotheses"]
            )
            print(f"Available datasets: {', '.join(sorted(available))}")
            return 1
        print(f"Filtered to {len(hypothesis_bank)} hypotheses for dataset '{args.dataset}'")

    # Filter by tier if specified
    if args.tiers:
        try:
            requested_tiers = _parse_tier_filter_values(args.tiers)
        except ValueError as exc:
            print(str(exc))
            return 1

        hypothesis_bank = [
            h for h in hypothesis_bank
            if str((h.get("complexity") or {}).get("tier", "")).upper() in requested_tiers
        ]
        if not hypothesis_bank:
            available_tiers = sorted(
                {
                    str((h.get("complexity") or {}).get("tier", "")).upper()
                    for h in bank["hypotheses"]
                    if (h.get("complexity") or {}).get("tier")
                }
            )
            print(f"No hypotheses found for requested tiers: {sorted(requested_tiers)}")
            if available_tiers:
                print(f"Available tiers in bank: {available_tiers}")
            return 1
        print(f"Filtered to {len(hypothesis_bank)} hypotheses for tiers {sorted(requested_tiers)}")

    # Determine base config
    if args.config:
        base_config = args.config
    else:
        base_config = repo_root / "src" / "veritas" / "workflow" / "configs" / "default.json"

    if not base_config.exists():
        print(f"Base config not found: {base_config}")
        return 1

    # Determine run parameters
    n_runs = 1 if args.pilot else args.n_runs
    hypotheses_filter = args.hypotheses
    context_length_override = _resolve_context_length_override(args.context_length)
    if args.context_length is not None and context_length_override is None:
        print("Invalid --context-length. Must be a positive integer.")
        return 1
    if context_length_override is not None:
        os.environ[GLOBAL_CONTEXT_ENV] = str(context_length_override)
        print(f"Using global context length override: {context_length_override}")

    numeric_selection: List[int] = []
    if args.hypothesis_range:
        try:
            numeric_selection = _parse_hypothesis_number_spec(args.hypothesis_range)
        except ValueError:
            print(
                "Invalid --hypothesis-range format. "
                "Use values like '1-8' or '1-4,7,10'."
            )
            return 1
    elif args.hypothesis_numbers:
        numeric_selection = sorted(set(int(n) for n in args.hypothesis_numbers))

    if numeric_selection:
        hypotheses_filter = _resolve_hypothesis_filter_ids(hypothesis_bank, numeric_selection)
        if not hypotheses_filter:
            available_numbers = sorted(
                {
                    n for n in
                    (_extract_hypothesis_number(h.get("id", "")) for h in hypothesis_bank)
                    if n is not None
                }
            )
            print("No hypotheses matched the requested numeric IDs/range.")
            if available_numbers:
                print(f"Available numbers in current selection: {available_numbers}")
            return 1
        print(
            f"Filtered by numeric hypothesis selection to {len(hypotheses_filter)} IDs "
            f"(numbers: {numeric_selection})"
        )

    # Create output directory — exact path takes priority over auto-timestamped subdirectory
    if args.exact_output_dir:
        output_dir = args.exact_output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"exp_{timestamp}"
        if args.tag:
            dir_name += f"_{args.tag}"
        output_dir = args.output_dir / dir_name
    eval_sesoi_override = args.eval_sesoi_value if args.eval_sesoi_value is not None else args.eval_sesoi_profile
    enable_critic = not args.disable_critic

    # Run experiments
    results = run_experiments(
        hypothesis_bank,
        base_config,
        output_dir,
        n_runs=n_runs,
        run_start_index=args.run_start_index,
        hypotheses_filter=hypotheses_filter,
        timeout=args.timeout,
        verbose=not args.quiet,
        dataset_registry=dataset_registry,
        use_ground_truth=args.use_ground_truth,
        eval_sesoi_override=eval_sesoi_override,
        context_length_override=context_length_override,
        enable_critic=enable_critic,
        max_transient_retries=max(0, args.max_transient_retries),
    )

    # Save results
    results_file = output_dir / "experiment_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    total = len(results)
    success = sum(1 for r in results if r["success"])
    correct = sum(1 for r in results if r.get("is_correct"))
    failed = total - success

    print(f"Total runs: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")

    if success > 0:
        valid = [r for r in results if r.get("is_correct") is not None]
        accuracy = correct / len(valid) if valid else 0
        print(f"Correct verdicts: {correct}/{len(valid)} ({accuracy:.1%})")

    # Verdict distribution
    verdicts = {}
    for r in results:
        v = r.get("verdict", "N/A")
        verdicts[v] = verdicts.get(v, 0) + 1
    print(f"\nVerdicts: {verdicts}")

    # Per-hypothesis breakdown
    print("\nPer-hypothesis results:")
    by_hypothesis = {}
    for r in results:
        hid = r["hypothesis_id"]
        if hid not in by_hypothesis:
            by_hypothesis[hid] = []
        by_hypothesis[hid].append(r)

    for hid, runs in by_hypothesis.items():
        verdicts_list = [r.get("verdict", "?") for r in runs]
        expected = runs[0].get("expected", "?")
        correct_count = sum(1 for r in runs if r.get("is_correct"))
        print(f"  {hid}:")
        print(f"    Expected: {expected}")
        print(f"    Verdicts: {verdicts_list}")
        print(f"    Correct: {correct_count}/{len(runs)}")

    if args.evaluate:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        if hypotheses_filter:
            eval_bank = [h for h in hypothesis_bank if h["id"] in hypotheses_filter]
        else:
            eval_bank = hypothesis_bank  # already filtered by --dataset if set
        eval_output = args.evaluation_output or (output_dir / "evaluation_results.json")
        verdicts = evaluate_batch(
            output_dir,
            eval_bank,
            dataset_registry=dataset_registry,
            output_file=eval_output,
            sesoi_override=eval_sesoi_override,
        )
        metrics = compute_metrics(verdicts)
        print_summary(verdicts, metrics)
        print(f"\nEvaluation saved to: {eval_output}")

    return 0


if __name__ == "__main__":
    exit(main())
