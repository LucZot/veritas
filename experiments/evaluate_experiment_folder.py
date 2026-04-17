#!/usr/bin/env python3
"""Evaluate an experiment output folder with automatic hypothesis-bank resolution.

Example:
  python experiments/evaluate_experiment_folder.py results/experiments/exp_20260219_124348
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation import evaluate_batch, compute_metrics, print_summary


def _load_bank(bank_path: Path) -> Dict[str, Any]:
    with open(bank_path) as f:
        bank = json.load(f)
    if "hypotheses" not in bank or not isinstance(bank["hypotheses"], list):
        raise ValueError(f"Invalid hypothesis bank format: {bank_path}")
    return bank


def _discover_hypothesis_dirs(results_dir: Path) -> List[str]:
    """Return hypothesis directory names under an experiment folder."""
    ids: List[str] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        has_run_dirs = any(grandchild.is_dir() for grandchild in child.glob("run_*"))
        if has_run_dirs:
            ids.append(child.name)
    return ids


def _score_bank_for_ids(bank_ids: set[str], present_ids: set[str]) -> Tuple[bool, int]:
    """Return (covers_all_present, overlap_count)."""
    covers_all = present_ids.issubset(bank_ids)
    overlap = len(bank_ids.intersection(present_ids))
    return covers_all, overlap


def _auto_select_bank(results_dir: Path, repo_root: Path) -> Tuple[Path, Dict[str, Any], List[str]]:
    present_ids = set(_discover_hypothesis_dirs(results_dir))
    candidate_paths = [
        repo_root / "experiments" / "tiered_hypothesis_bank.json",
    ]

    candidates: List[Tuple[Path, Dict[str, Any], bool, int]] = []
    for bank_path in candidate_paths:
        if not bank_path.exists():
            continue
        bank = _load_bank(bank_path)
        ids = {str(h.get("id", "")) for h in bank["hypotheses"]}
        covers_all, overlap = _score_bank_for_ids(ids, present_ids)
        candidates.append((bank_path, bank, covers_all, overlap))

    if not candidates:
        raise FileNotFoundError("No hypothesis bank candidates found in experiments/")

    full_cover = [c for c in candidates if c[2]]
    if full_cover:
        selected_path, selected_bank, _, _ = min(full_cover, key=lambda x: len(x[1]["hypotheses"]))
    else:
        selected_path, selected_bank, _, _ = max(candidates, key=lambda x: x[3])

    selected_ids = _discover_hypothesis_dirs(results_dir)
    return selected_path, selected_bank, selected_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a full experiment folder and compute aggregate metrics."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to experiment folder (e.g., results/experiments/exp_20260219_124348)",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        help="Optional hypothesis bank path. If omitted, script auto-selects from experiments/*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for per-run evaluation results (default: <results_dir>/evaluation_results.json)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional output path for aggregate metrics JSON (default: <results_dir>/evaluation_metrics.json)",
    )
    sesoi_group = parser.add_mutually_exclusive_group()
    sesoi_group.add_argument(
        "--sesoi-profile",
        choices=["strict", "standard", "loose"],
        help="Override SESOI profile for all hypotheses in this evaluation",
    )
    sesoi_group.add_argument(
        "--sesoi-value",
        type=float,
        help="Override SESOI with a fixed numeric value for all test families",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    repo_root = Path(__file__).resolve().parent.parent
    if args.bank:
        bank_path = args.bank.resolve()
        if not bank_path.exists():
            raise FileNotFoundError(f"Hypothesis bank not found: {bank_path}")
        bank = _load_bank(bank_path)
        selected_ids = _discover_hypothesis_dirs(results_dir)
    else:
        bank_path, bank, selected_ids = _auto_select_bank(results_dir, repo_root)

    if not selected_ids:
        raise RuntimeError(f"No hypothesis run directories found under: {results_dir}")

    selected_set = set(selected_ids)
    eval_hypotheses = [h for h in bank["hypotheses"] if str(h.get("id", "")) in selected_set]
    if not eval_hypotheses:
        raise RuntimeError(
            f"No matching hypotheses in bank {bank_path} for runs in {results_dir}"
        )

    missing = sorted(selected_set - {str(h.get("id", "")) for h in eval_hypotheses})
    if missing:
        print(f"Warning: {len(missing)} run directories not found in selected bank: {missing}")

    output_file = args.output or (results_dir / "evaluation_results.json")
    metrics_file = args.metrics_output or (results_dir / "evaluation_metrics.json")
    sesoi_override = args.sesoi_value if args.sesoi_value is not None else args.sesoi_profile

    print(f"Results dir: {results_dir}")
    print(f"Hypothesis bank: {bank_path}")
    print(f"Matched hypotheses: {len(eval_hypotheses)}")
    print(f"Per-run output: {output_file}")
    print(f"Metrics output: {metrics_file}")

    verdicts = evaluate_batch(
        results_dir=results_dir,
        hypothesis_bank=eval_hypotheses,
        dataset_registry=bank.get("dataset_registry"),
        output_file=output_file,
        sesoi_override=sesoi_override,
    )
    metrics = compute_metrics(verdicts)
    print_summary(verdicts, metrics)

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics JSON to: {metrics_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
