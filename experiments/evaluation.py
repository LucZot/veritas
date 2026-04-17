"""
Evaluation framework for hypothesis testing workflows.

Extracts structured verdicts from workflow outputs and compares to ground truth.
"""

import json
import re
import math
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class WorkflowVerdict:
    """Structured verdict from a workflow run."""

    # Run metadata
    hypothesis_id: str
    run_id: str
    output_dir: str

    # Verdict (YES, NO, INCONCLUSIVE, or PARSE_ERROR)
    verdict: str

    # Statistics extracted from output (optional)
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    test_used: Optional[str] = None
    sample_sizes: Optional[Dict[str, Any]] = None
    n_total: Optional[int] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None
    evidence_label_reported: Optional[str] = None

    # Ground truth comparison
    expected_conclusion: Optional[str] = None
    is_correct: Optional[bool] = None
    expected_label: Optional[str] = None
    is_label_correct: Optional[bool] = None
    expected_invalid_subtype: Optional[str] = None

    # Hypothesis metadata (for stratified reporting)
    complexity_tier: Optional[str] = None
    control_type: Optional[str] = None
    control_category: Optional[str] = None

    # Evidence label (v2)
    evidence_label_raw: Optional[str] = None  # computed from Phase 2B before integrity overrides
    evidence_label: Optional[str] = None
    power_at_sesoi: Optional[float] = None
    sesoi_value: Optional[float] = None
    sesoi_profile: Optional[str] = None
    test_family: Optional[str] = None
    direction_intent: Optional[str] = None
    direction_groups: Optional[List[str]] = None

    # Execution metadata
    completed_all_phases: bool = False
    parse_method: str = "unknown"  # "config", "json_block", "regex", "failed"
    parse_error: Optional[str] = None
    failure_code: Optional[str] = None
    failure_stage: Optional[str] = None
    phase1_feasibility_status: Optional[str] = None
    phase1_invalid_subtype: Optional[str] = None
    early_stop_type: Optional[str] = None  # e.g., "phase1_untestable"

    # Phase 2B alignment checks
    phase2b_consistent: Optional[bool] = None
    phase2b_mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Dataset limitation flag (hypothesis bank ground_truth.dataset_limitation)
    has_dataset_limitation: bool = False

    # Coding trial counts (from phase execution metadata)
    coding_trials_phase2a: Optional[int] = None  # total code executions in Phase 2A
    coding_trials_phase2b: Optional[int] = None  # total code executions in Phase 2B
    coding_trials_total: Optional[int] = None    # sum across all phases

    # Data curation / hallucination diagnostics
    phase2a_unique_patients: Optional[int] = None
    phase2b_sample_coverage: Optional[float] = None
    phase2b_group_subset_violation: Optional[bool] = None
    phase2b_missing_planned_groups: List[str] = field(default_factory=list)
    phase2b_n_total_sum_mismatch: Optional[bool] = None
    phase2b_literal_pvalue_assignment: Optional[bool] = None
    phase2b_boundary_pvalue_005: Optional[bool] = None


def _load_phase2b_results(output_dir: Path) -> Optional[Dict[str, Any]]:
    results_path = output_dir / "phase2b_statistical_analysis" / "workspace" / "data" / "statistical_results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_coding_trials(output_dir: Path) -> Dict[str, Optional[int]]:
    """Load code execution trial counts from phase execution metadata files."""
    phase_dirs = {
        "phase2a": output_dir / "phase2a_imaging_analysis",
        "phase2b": output_dir / "phase2b_statistical_analysis",
    }
    counts: Dict[str, Optional[int]] = {}
    for phase_key, phase_dir in phase_dirs.items():
        meta_path = phase_dir / "discussion_execution_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                counts[phase_key] = meta.get("total_executions")
            except Exception:
                counts[phase_key] = None
        else:
            counts[phase_key] = None
    return counts


def _resolve_phase2b_code_path(workspace_dir: Path, filename: str) -> Optional[Path]:
    try:
        raw_path = Path(str(filename).strip())
        candidate = raw_path if raw_path.is_absolute() else workspace_dir / raw_path
        resolved = candidate.resolve()
        resolved.relative_to(workspace_dir.resolve())
    except Exception:
        return None
    if resolved.suffix != ".py" or not resolved.exists():
        return None
    return resolved


def _get_phase2b_active_code_files(output_dir: Path) -> List[Path]:
    workspace_dir = output_dir / "phase2b_statistical_analysis" / "workspace"
    code_dir = workspace_dir / "code"
    default_files = sorted(code_dir.rglob("*.py")) if code_dir.exists() else []
    if not default_files:
        return []

    metadata_path = output_dir / "phase2b_statistical_analysis" / "discussion_execution_metadata.json"
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
    return [selected_path]


def _detect_synthetic_phase2b_usage(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Best-effort detection of synthetic/mock data generation in Phase 2B code artifacts."""
    files_to_scan = _get_phase2b_active_code_files(output_dir)
    if not files_to_scan:
        return None

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

    hits: List[Dict[str, Any]] = []
    soft_hits: List[Dict[str, Any]] = []
    for py_file in files_to_scan:
        try:
            lines = py_file.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        try:
            rel = str(py_file.relative_to(output_dir.resolve()))
        except Exception:
            rel = str(py_file)
        for lineno, line in enumerate(lines, start=1):
            lowered = line.lower()
            window = " ".join(
                l.lower() for l in lines[max(0, lineno - 3): min(len(lines), lineno + 2)]
            )
            for rule_name, phrase in phrase_rules:
                if phrase.search(lowered):
                    hits.append({"file": rel, "line": lineno, "rule": rule_name})

            if "placeholder" in lowered:
                benign_tokens = ("p-value", "p values", "confidence interval", "ci_", "ci ", "effect size", "for each group")
                hard_tokens = ("calculation", "derive", "derived", "measurement", "metric", "feature", "gls", "volume", "data")
                if any(tok in window for tok in benign_tokens):
                    soft_hits.append({"file": rel, "line": lineno, "rule": "placeholder_nonblocking"})
                elif any(tok in window for tok in hard_tokens):
                    hits.append({"file": rel, "line": lineno, "rule": "placeholder_analysis"})
                else:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "placeholder_nonblocking"})

            np_match = np_random_any.search(line)
            if np_match:
                fn = np_match.group(1).lower()
                jitter_ctx = ("jitter" in window) or (("scatter(" in window or "plt.scatter" in window) and "np.random.normal" in lowered)
                resample_ctx = any(tok in window for tok in ("bootstrap", "resampl", "permutation", "permute", "shuffle"))
                if fn in allowed_np_random:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_nonblocking"})
                elif jitter_ctx:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_plot_jitter"})
                elif resample_ctx and fn == "choice":
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_resampling"})
                else:
                    hits.append({"file": rel, "line": lineno, "rule": "np_random_distribution"})

            py_match = py_random_any.search(line)
            if py_match:
                fn = py_match.group(1).lower()
                if fn in allowed_py_random:
                    soft_hits.append({"file": rel, "line": lineno, "rule": "rng_nonblocking"})
                else:
                    hits.append({"file": rel, "line": lineno, "rule": "py_random_distribution"})

            if torch_random_any.search(line):
                hits.append({"file": rel, "line": lineno, "rule": "torch_random_distribution"})
    if not hits and not soft_hits:
        return None
    return {
        "count": len(hits),
        "hits": hits[:5],
        "soft_count": len(soft_hits),
        "soft_hits": soft_hits[:5],
    }


def _load_phase1_feasibility(output_dir: Path) -> Dict[str, Optional[str]]:
    """Best-effort load of Phase 1 feasibility decision from workflow config."""
    config_path = output_dir / "workflow_config.json"
    if not config_path.exists():
        return {"status": None, "invalid_subtype": None}
    try:
        with open(config_path) as f:
            wf_config = json.load(f)
        plan = wf_config.get("plan") or {}
        feasibility = plan.get("feasibility") or {}
        status = feasibility.get("status")
        invalid_subtype = feasibility.get("invalid_subtype")
        if isinstance(status, str):
            status = status.upper()
        return {"status": status, "invalid_subtype": invalid_subtype}
    except Exception:
        return {"status": None, "invalid_subtype": None}


def _load_workflow_audit(output_dir: Path) -> Dict[str, Any]:
    """Best-effort load of workflow_audit from workflow_config.json."""
    config_path = output_dir / "workflow_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            wf_config = json.load(f)
        audit = wf_config.get("workflow_audit")
        return audit if isinstance(audit, dict) else {}
    except Exception:
        return {}


def _normalize_test_name(name: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# Canonical test family mappings for fuzzy matching
_TEST_FAMILY_ALIASES: Dict[str, List[str]] = {
    "group_difference": [
        "t-test", "ttest", "t test", "independent t-test", "independent t test",
        "welch", "student", "mann-whitney", "mann whitney", "mannwhitney",
        "wilcoxon rank-sum", "wilcoxon rank sum", "mwu", "anova", "kruskal",
    ],
    "correlation": [
        "pearson", "spearman", "kendall", "correlation", "corr",
        "pearson correlation", "spearman correlation",
        "pearson's r", "spearman's rho",
    ],
    "regression": [
        "regression", "linear regression", "ols", "logistic regression",
        "partial correlation",
    ],
    "survival": [
        "survival", "log-rank", "log rank", "logrank", "kaplan", "cox",
        "coxph", "cox proportional hazards", "hazard ratio",
    ],
}


def _test_names_same_family(name_a: str, name_b: str) -> bool:
    """Check if two test names belong to the same statistical family."""
    a = _normalize_test_name(name_a)
    b = _normalize_test_name(name_b)
    if a == b:
        return True
    family_a = _infer_test_family_from_name(a)
    family_b = _infer_test_family_from_name(b)
    if family_a and family_b:
        return family_a == family_b
    return False


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF for standard normal (Acklam)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    # Coefficients in rational approximations
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _get_sample_sizes(sample_sizes: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not isinstance(sample_sizes, dict):
        return None, None, None
    counts = []
    for _, v in sample_sizes.items():
        try:
            counts.append(int(v))
        except Exception:
            return None, None, None
    if not counts:
        return None, None, None
    n_total = sum(counts)
    if len(counts) >= 2:
        return counts[0], counts[1], n_total
    return counts[0], None, n_total


def _infer_test_family_from_name(test_name: str) -> Optional[str]:
    name = _normalize_test_name(test_name)
    if not name:
        return None
    # Check against known aliases for each family
    for family, aliases in _TEST_FAMILY_ALIASES.items():
        for alias in aliases:
            if alias in name:
                return family
    # Fallback substring checks
    if "corr" in name or "spearman" in name or "pearson" in name:
        return "correlation"
    if "regress" in name or "ols" in name:
        return "regression"
    if "survival" in name or "logrank" in name or "log-rank" in name or "cox" in name:
        return "survival"
    if "t-test" in name or "ttest" in name or "mann" in name or "whitney" in name or "anova" in name:
        return "group_difference"
    return None


def _resolve_test_family(hypothesis: Dict[str, Any], phase2b: Optional[Dict[str, Any]]) -> Optional[str]:
    analysis_intent = hypothesis.get("analysis_intent") if hypothesis else None
    if isinstance(analysis_intent, dict):
        analysis_type = str(analysis_intent.get("analysis_type") or "").strip().lower()
        if analysis_type == "survival":
            return "survival"

    meta = hypothesis.get("meta_analysis") if hypothesis else None
    if isinstance(meta, dict) and meta.get("test_family"):
        return meta.get("test_family")
    if phase2b and phase2b.get("analysis_type"):
        return str(phase2b.get("analysis_type"))
    if phase2b and phase2b.get("test_performed"):
        return _infer_test_family_from_name(str(phase2b.get("test_performed")))
    return None


def _resolve_direction_intent(hypothesis: Dict[str, Any]) -> str:
    meta = hypothesis.get("meta_analysis") if hypothesis else None
    if isinstance(meta, dict) and meta.get("direction_intent"):
        return str(meta.get("direction_intent"))
    return "non_directional"


def _resolve_direction_groups(hypothesis: Dict[str, Any]) -> Optional[List[str]]:
    meta = hypothesis.get("meta_analysis") if hypothesis else None
    if isinstance(meta, dict) and meta.get("direction_groups"):
        groups = meta.get("direction_groups")
        if isinstance(groups, list) and len(groups) >= 2:
            return [str(groups[0]), str(groups[1])]
    return None


def _normalize_constraint_level(level: Optional[Any]) -> str:
    value = str(level or "none").strip().lower()
    if value in {"required", "recommended", "none"}:
        return value
    return "none"


def _get_analysis_constraint(hypothesis: Dict[str, Any], key: str) -> Dict[str, Any]:
    constraints = hypothesis.get("analysis_constraints") if hypothesis else None
    if not isinstance(constraints, dict):
        return {"level": "none"}
    entry = constraints.get(key)
    if isinstance(entry, str):
        return {"level": _normalize_constraint_level(entry)}
    if isinstance(entry, dict):
        payload = dict(entry)
        payload["level"] = _normalize_constraint_level(payload.get("level"))
        return payload
    return {"level": "none"}


def _normalize_field_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    normalized = []
    for item in values:
        token = str(item).strip()
        if token:
            normalized.append(token)
    return normalized


def _resolve_sesoi_profile(hypothesis: Dict[str, Any], dataset_registry: Optional[Dict[str, Any]]):
    """Resolve SESOI profile. Returns a string name or a custom dict."""
    if hypothesis and hypothesis.get("sesoi_profile"):
        profile = hypothesis.get("sesoi_profile")
        # Support custom dict profiles (e.g., {"cohen_d": 0.3})
        if isinstance(profile, dict):
            return profile
        return str(profile)
    dataset_name = None
    if hypothesis:
        dataset = hypothesis.get("dataset") or {}
        dataset_name = dataset.get("name")
    if dataset_registry and dataset_name and dataset_name in dataset_registry:
        default_profile = dataset_registry[dataset_name].get("default_sesoi_profile")
        if default_profile:
            if isinstance(default_profile, dict):
                return default_profile
            return str(default_profile)
    return "standard"


def _sesoi_value(profile, test_family: Optional[str]) -> Optional[float]:
    """Resolve SESOI value from a named profile or custom dict.

    Args:
        profile: Either a string ("strict"/"standard"/"loose") or a dict
                 with a direct numeric value (e.g., {"cohen_d": 0.3}).
        test_family: The statistical test family.
    """
    # Support direct numeric SESOI override
    if isinstance(profile, (int, float)):
        try:
            return float(profile)
        except (TypeError, ValueError):
            return None

    # Support custom numeric SESOI from config
    if isinstance(profile, dict):
        # Accept any numeric value key (cohen_d, r, effect_size, etc.)
        for key in ("cohen_d", "r", "effect_size", "value"):
            if key in profile:
                try:
                    return float(profile[key])
                except (TypeError, ValueError):
                    pass
        return None

    profile_str = (str(profile) if profile else "standard").lower()
    table = {
        "strict": {"group_difference": 0.2, "correlation": 0.2, "regression": 0.2, "survival": 1.2},
        "standard": {"group_difference": 0.5, "correlation": 0.3, "regression": 0.3, "survival": 1.5},
        "loose": {"group_difference": 0.8, "correlation": 0.4, "regression": 0.4, "survival": 2.0},
    }
    return table.get(profile_str, table["standard"]).get(test_family)


def _power_ttest_ind(d: float, n1: int, n2: int, alpha: float = 0.05) -> Optional[float]:
    if n1 <= 1 or n2 <= 1:
        return None
    n_eff = (n1 * n2) / (n1 + n2)
    z_alpha = _norm_ppf(1 - alpha / 2)
    ncp = d * math.sqrt(n_eff)
    return _norm_cdf(-z_alpha - ncp) + (1 - _norm_cdf(z_alpha - ncp))


def _power_corr(rho: float, n: int, alpha: float = 0.05) -> Optional[float]:
    if n <= 3:
        return None
    rho = max(min(rho, 0.999999), -0.999999)
    z = math.atanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    z_alpha = _norm_ppf(1 - alpha / 2)
    ncp = z / se
    return _norm_cdf(-z_alpha - ncp) + (1 - _norm_cdf(z_alpha - ncp))


def _power_survival_hr(
    hr: float,
    n_events: int,
    allocation: float = 0.5,
    alpha: float = 0.05,
) -> Optional[float]:
    """Approximate survival power via Schoenfeld's event-based formula."""
    if hr is None or hr <= 0 or n_events <= 0:
        return None
    if allocation <= 0 or allocation >= 1:
        allocation = 0.5
    log_hr = abs(math.log(hr))
    if log_hr <= 0:
        return None
    z_alpha = _norm_ppf(1 - alpha / 2)
    info = n_events * allocation * (1 - allocation) * (log_hr ** 2)
    if info <= 0:
        return None
    z_beta = math.sqrt(info) - z_alpha
    return _norm_cdf(z_beta)


def _extract_survival_event_count(phase2b: Optional[Dict[str, Any]], n_total: Optional[int]) -> Optional[int]:
    if not isinstance(phase2b, dict):
        return n_total if n_total and n_total > 0 else None

    direct_keys = ("n_events", "events", "events_observed", "event_count", "num_events")
    for key in direct_keys:
        value = _safe_float(phase2b.get(key))
        if value is not None and value > 0:
            return int(value)

    event_rate = _safe_float(phase2b.get("event_rate"))
    if event_rate is not None and n_total and 0 < event_rate <= 1:
        return max(1, int(round(event_rate * n_total)))

    stats = phase2b.get("group_statistics")
    if isinstance(stats, dict):
        total_events = 0
        found = False
        for payload in stats.values():
            if not isinstance(payload, dict):
                continue
            for key in ("events", "event_count", "n_events"):
                value = _safe_float(payload.get(key))
                if value is not None and value >= 0:
                    total_events += int(value)
                    found = True
                    break
        if found and total_events > 0:
            return total_events

    return n_total if n_total and n_total > 0 else None


def _compute_power_at_sesoi(
    test_family: Optional[str],
    sesoi: Optional[float],
    n1: Optional[int],
    n2: Optional[int],
    n_total: Optional[int],
    phase2b: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    if sesoi is None or test_family is None:
        return None
    if test_family == "group_difference":
        if n1 is None or n2 is None:
            return None
        return _power_ttest_ind(abs(sesoi), n1, n2)
    if test_family in {"correlation", "regression"}:
        n = n_total or (n1 + n2 if n1 and n2 else None)
        if not n:
            return None
        return _power_corr(abs(sesoi), n)
    if test_family == "survival":
        if sesoi <= 0:
            return None
        hr = float(sesoi)
        allocation = 0.5
        if n1 and n2 and (n1 + n2) > 0:
            allocation = n1 / (n1 + n2)
        n_events = _extract_survival_event_count(phase2b, n_total)
        if not n_events:
            return None
        return _power_survival_hr(hr=hr, n_events=int(n_events), allocation=allocation)
    return None


def _reference_ci(effect_size: float, test_family: Optional[str], n1: Optional[int], n2: Optional[int], n_total: Optional[int]) -> Optional[Tuple[float, float]]:
    if effect_size is None or test_family is None:
        return None
    if test_family == "group_difference" and n1 and n2:
        n_total_val = n1 + n2
        if n_total_val <= 2:
            return None
        # Approximate SE for Cohen's d
        se = math.sqrt((n1 + n2) / (n1 * n2) + (effect_size ** 2) / (2 * (n_total_val - 2)))
        return effect_size - 1.96 * se, effect_size + 1.96 * se
    if test_family in {"correlation", "regression"} and n_total and n_total > 3:
        r = max(min(effect_size, 0.999999), -0.999999)
        z = math.atanh(r)
        se = 1.0 / math.sqrt(n_total - 3)
        z_low = z - 1.96 * se
        z_high = z + 1.96 * se
        return math.tanh(z_low), math.tanh(z_high)
    return None


def _is_mwu_test(phase2b: Dict[str, Any]) -> bool:
    """Check if the test is Mann-Whitney U (nonparametric)."""
    test_name = str(phase2b.get("test_performed", "")).lower()
    effect_type = str(phase2b.get("effect_size_type", "")).lower().replace("-", "_")
    return ("mann" in test_name and "whitney" in test_name) or effect_type == "rank_biserial"


def _ci_checks(phase2b: Dict[str, Any], test_family: Optional[str], n1: Optional[int], n2: Optional[int], n_total: Optional[int]) -> List[str]:
    warnings: List[str] = []
    is_mwu = _is_mwu_test(phase2b)
    ci = phase2b.get("confidence_interval") if isinstance(phase2b, dict) else None
    if not isinstance(ci, dict):
        # MWU tests may legitimately omit CI (bootstrap CI is complex)
        if not is_mwu:
            warnings.append("ci_missing")
        return warnings

    lower = _safe_float(ci.get("lower"))
    upper = _safe_float(ci.get("upper"))
    if lower is None or upper is None:
        # Null CI is acceptable for MWU (parametric CI inappropriate)
        if not is_mwu:
            warnings.append("ci_not_numeric")
        return warnings
    if lower > upper:
        warnings.append("ci_bounds_inverted")

    ci_type = None
    if isinstance(phase2b.get("confidence_interval_type"), str):
        ci_type = phase2b.get("confidence_interval_type").strip().lower()
    if ci_type not in {"effect_size", "mean_difference"}:
        ci_type = None

    effect_size = _safe_float(phase2b.get("effect_size"))
    effect_size_type = str(phase2b.get("effect_size_type", "")).lower().replace("-", "_")
    if ci_type is None and test_family == "group_difference":
        warnings.append("ci_type_not_declared")

    if ci_type == "mean_difference":
        groups = _groups_from_phase2b(phase2b)
        if groups and len(groups) >= 2:
            diff = _compute_group_mean_diff(phase2b, groups)
            if diff is not None and not (lower <= diff <= upper):
                reverse_diff = _compute_group_mean_diff(phase2b, [groups[1], groups[0]])
                if reverse_diff is None or not (lower <= reverse_diff <= upper):
                    warnings.append("ci_does_not_contain_mean_diff")
    else:
        if effect_size is not None and not (lower <= effect_size <= upper):
            if test_family == "survival" and effect_size_type == "hazard_ratio":
                # Survival CI is often reported on log-HR scale while effect_size stores HR.
                # Skip strict containment to avoid systematic false warnings.
                pass
            else:
                warnings.append("ci_does_not_contain_effect")

    if test_family in {"correlation", "regression"} and ci_type != "mean_difference":
        if lower < -1 or upper > 1:
            warnings.append("ci_out_of_bounds")

    if ci_type != "mean_difference":
        ref = _reference_ci(effect_size, test_family, n1, n2, n_total)
        if ref:
            ref_width = ref[1] - ref[0]
            agent_width = upper - lower
            if ref_width > 0 and agent_width < 0.25 * ref_width:
                warnings.append("ci_too_narrow")

    return warnings


def _load_plan_groups(output_dir: Path) -> Optional[List[str]]:
    config_path = output_dir / "workflow_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        plan = config.get("plan", {})
        groups = plan.get("groups")
        if isinstance(groups, list) and len(groups) >= 2:
            return [str(groups[0]), str(groups[1])]
    except Exception:
        return None
    return None


def _load_plan_statistical_test(output_dir: Path) -> Optional[str]:
    config_path = output_dir / "workflow_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        plan = config.get("plan", {})
        test_name = plan.get("statistical_test")
        if isinstance(test_name, str) and test_name.strip():
            return test_name.strip()
    except Exception:
        return None
    return None


def _compute_group_mean_diff(phase2b: Dict[str, Any], groups: List[str]) -> Optional[float]:
    stats = phase2b.get("group_statistics") if isinstance(phase2b, dict) else None
    if not isinstance(stats, dict):
        return None
    group_a, group_b = groups[0], groups[1]
    if group_a not in stats or group_b not in stats:
        return None
    mean_a = _safe_float(stats[group_a].get("mean")) if isinstance(stats[group_a], dict) else None
    mean_b = _safe_float(stats[group_b].get("mean")) if isinstance(stats[group_b], dict) else None
    if mean_a is None or mean_b is None:
        return None
    return mean_a - mean_b


def _groups_from_phase2b(phase2b: Dict[str, Any]) -> Optional[List[str]]:
    if not isinstance(phase2b, dict):
        return None
    ci_groups = phase2b.get("confidence_interval_groups")
    if isinstance(ci_groups, list) and len(ci_groups) >= 2:
        return [str(ci_groups[0]), str(ci_groups[1])]
    sample_sizes = phase2b.get("sample_sizes")
    if isinstance(sample_sizes, dict) and len(sample_sizes) >= 2:
        return list(sample_sizes.keys())
    stats = phase2b.get("group_statistics")
    if isinstance(stats, dict) and len(stats) >= 2:
        return list(stats.keys())
    return None


def _infer_observed_direction(
    phase2b: Optional[Dict[str, Any]],
    direction_groups: Optional[List[str]],
) -> Optional[float]:
    if not phase2b:
        return None
    if direction_groups and len(direction_groups) >= 2:
        diff = _compute_group_mean_diff(phase2b, direction_groups)
        if diff is not None:
            return diff
    effect_size = _safe_float(phase2b.get("effect_size"))
    return effect_size


def _infer_survival_direction(phase2b: Optional[Dict[str, Any]]) -> Optional[float]:
    """Return a signed direction proxy for survival effects.

    Positive => improved survival, Negative => worse survival.
    """
    if not phase2b:
        return None
    effect_size = _safe_float(phase2b.get("effect_size"))
    if effect_size is None:
        return None

    effect_size_type = str(phase2b.get("effect_size_type", "")).lower().replace("-", "_")
    if effect_size_type == "hazard_ratio":
        if effect_size <= 0:
            return None
        return -math.log(effect_size)
    return effect_size



def _requires_group_adjustment(output_dir: Optional[Path], analysis_type: Optional[str]) -> bool:
    """Check if a correlation/regression across mixed groups needs group adjustment."""
    if analysis_type not in {"correlation", "regression"}:
        return False
    if not output_dir:
        return False
    config_path = Path(output_dir) / "workflow_config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        return False
    plan = config.get("plan", {})
    cohort_mode = str(plan.get("cohort_mode", "")).strip().lower()
    groups = plan.get("groups") or []
    all_groups = cohort_mode == "all" or (len(groups) == 1 and str(groups[0]).strip().lower() == "all")
    if not all_groups:
        return False
    # Check if dataset has multiple groups (available_groups in registry)
    dataset = config.get("dataset", {})
    available_groups = dataset.get("available_groups", [])
    return len(available_groups) >= 2


def _has_group_adjustment(phase2b: Optional[Dict[str, Any]], output_dir: Optional[Path] = None) -> bool:
    """Check if the analysis adjusted or stratified by group."""
    if not phase2b:
        return False

    # Accept "group" and any legacy field names that represent the group variable
    group_terms = {"group"}

    # Dynamically detect group field from workflow config if available
    if output_dir:
        config_path = Path(output_dir) / "workflow_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                group_field = config.get("dataset", {}).get("group_field")
                if group_field:
                    group_terms.add(str(group_field).strip().lower())
            except Exception:
                pass

    adjusted_for = phase2b.get("adjusted_for")
    if isinstance(adjusted_for, str) and adjusted_for.strip().lower() in group_terms:
        return _has_valid_adjusted_effect(phase2b)
    if isinstance(adjusted_for, list):
        for item in adjusted_for:
            if str(item).strip().lower() in group_terms:
                return _has_valid_adjusted_effect(phase2b)

    stratified_by = phase2b.get("stratified_by")
    has_group_strata = False
    if isinstance(stratified_by, str) and stratified_by.strip().lower() in group_terms:
        has_group_strata = True
    elif isinstance(stratified_by, list):
        for item in stratified_by:
            if str(item).strip().lower() in group_terms:
                has_group_strata = True
                break

    if has_group_strata:
        stratified_results = phase2b.get("stratified_results")
        return isinstance(stratified_results, dict) and len(stratified_results) > 0

    return False


def _has_valid_adjusted_effect(phase2b: Optional[Dict[str, Any]]) -> bool:
    if not phase2b:
        return False
    effect_size_type = str(phase2b.get("effect_size_type", "")).lower().replace("-", "_")
    adjusted_effect_types = {
        "partial_pearson_r",
        "partial_spearman_r",
        "regression_beta",
        "regression_beta_std",
        "regression_coef",
    }
    if effect_size_type in adjusted_effect_types:
        return True
    adjusted_block = phase2b.get("adjusted_effect_size")
    if isinstance(adjusted_block, dict):
        block_type = str(adjusted_block.get("effect_size_type", "")).lower().replace("-", "_")
        if block_type in adjusted_effect_types:
            return True
    return False


def compute_evidence_label(
    phase2b: Optional[Dict[str, Any]],
    hypothesis: Dict[str, Any],
    dataset_registry: Optional[Dict[str, Any]],
    output_dir: Optional[Path],
    sesoi_override: Optional[Any] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "evidence_label": None,
        "power_at_sesoi": None,
        "sesoi_value": None,
        "sesoi_profile": None,
        "test_family": None,
        "direction_intent": None,
        "direction_groups": None,
        "warnings": [],
        "invalid_reasons": [],
        "n_total": None,
        "invalid_subtype": None,
    }

    if not phase2b:
        result["evidence_label"] = "INVALID"
        result["invalid_reasons"].append("phase2b_results_missing")
        return result

    test_family = _resolve_test_family(hypothesis, phase2b)
    direction_intent = _resolve_direction_intent(hypothesis)
    direction_groups = _resolve_direction_groups(hypothesis)
    analysis_type = str(phase2b.get("analysis_type", "")).lower()

    n1, n2, n_total_from_sizes = _get_sample_sizes(phase2b.get("sample_sizes"))
    n_total = _safe_float(phase2b.get("n_total"))
    n_total_val = int(n_total) if n_total is not None else (n_total_from_sizes or None)

    result.update({
        "test_family": test_family,
        "direction_intent": direction_intent,
        "direction_groups": direction_groups,
        "n_total": n_total_val,
    })

    # Basic numeric validation
    p_value = _safe_float(phase2b.get("p_value"))
    effect_size = _safe_float(phase2b.get("effect_size"))
    if p_value is None or not (0 <= p_value <= 1):
        result["invalid_reasons"].append("p_value_invalid")
    if effect_size is None or not math.isfinite(effect_size):
        result["invalid_reasons"].append("effect_size_invalid")

    # Test family consistency (correlation ↔ regression are compatible)
    inferred_family = _infer_test_family_from_name(str(phase2b.get("test_performed", "")))
    _compatible = {frozenset({"correlation", "regression"})}
    if test_family and inferred_family and test_family != inferred_family:
        if frozenset({test_family, inferred_family}) not in _compatible:
            result["invalid_reasons"].append("test_family_mismatch")

    # CI checks
    result["warnings"].extend(_ci_checks(phase2b, test_family, n1, n2, n_total_val))

    # Constraint checks (required/recommended/none)
    test_family_constraint = _get_analysis_constraint(hypothesis, "test_family")
    expected_test_family = str(test_family_constraint.get("value") or "").strip().lower()
    if expected_test_family:
        if test_family != expected_test_family:
            if test_family_constraint.get("level") == "required":
                result["invalid_reasons"].append("test_family_constraint_violation")
            elif test_family_constraint.get("level") == "recommended":
                result["warnings"].append("test_family_constraint_recommended_not_met")

    adjust_for_constraint = _get_analysis_constraint(hypothesis, "adjust_for")
    required_adjust_fields = _normalize_field_list(adjust_for_constraint.get("fields"))
    if required_adjust_fields:
        reported_adjust_for = _normalize_field_list(phase2b.get("adjusted_for"))
        missing = [field for field in required_adjust_fields if field not in reported_adjust_for]
        if missing:
            if adjust_for_constraint.get("level") == "required":
                result["invalid_reasons"].append("adjust_for_constraint_violation")
            elif adjust_for_constraint.get("level") == "recommended":
                result["warnings"].append("adjust_for_constraint_recommended_not_met")

    stratify_constraint = _get_analysis_constraint(hypothesis, "stratify_by")
    required_stratify_fields = _normalize_field_list(stratify_constraint.get("fields"))
    if required_stratify_fields:
        reported_stratified_by = _normalize_field_list(phase2b.get("stratified_by"))
        missing = [field for field in required_stratify_fields if field not in reported_stratified_by]
        if missing:
            if stratify_constraint.get("level") == "required":
                result["invalid_reasons"].append("stratify_by_constraint_violation")
            elif stratify_constraint.get("level") == "recommended":
                result["warnings"].append("stratify_by_constraint_recommended_not_met")

    # Confounding check for mixed-group correlations (recommended by default, required if annotated)
    if _requires_group_adjustment(output_dir, analysis_type):
        group_adjustment_constraint = _get_analysis_constraint(hypothesis, "group_adjustment")
        level = group_adjustment_constraint.get("level") or "recommended"
        if level == "none":
            level = "recommended"
        if not _has_group_adjustment(phase2b, output_dir):
            if level == "required":
                result["invalid_reasons"].append("group_unadjusted")
            elif level == "recommended":
                result["warnings"].append("group_unadjusted_recommended")

    # SESOI + power
    sesoi_profile = sesoi_override if sesoi_override is not None else _resolve_sesoi_profile(hypothesis, dataset_registry)
    sesoi_value = _sesoi_value(sesoi_profile, test_family)
    power_at_sesoi = _compute_power_at_sesoi(
        test_family, sesoi_value, n1, n2, n_total_val, phase2b=phase2b
    )

    result.update({
        "sesoi_profile": sesoi_profile,
        "sesoi_value": sesoi_value,
        "power_at_sesoi": power_at_sesoi,
    })

    if result["invalid_reasons"]:
        result["evidence_label"] = "INVALID"
        return result

    # Directionality
    observed_direction = None
    if test_family == "survival":
        observed_direction = _infer_survival_direction(phase2b)
    elif test_family == "group_difference":
        groups = direction_groups or (_load_plan_groups(output_dir) if output_dir else None)
        if groups and len(groups) >= 2:
            diff = _compute_group_mean_diff(phase2b, groups)
            if diff is not None:
                observed_direction = diff
        if observed_direction is None:
            observed_direction = effect_size
    else:
        observed_direction = effect_size

    # Evidence label logic
    if p_value < 0.05:
        if direction_intent in {"positive", "negative"} and observed_direction is not None:
            if direction_intent == "positive" and observed_direction < 0:
                result["evidence_label"] = "REFUTED"
            elif direction_intent == "negative" and observed_direction > 0:
                result["evidence_label"] = "REFUTED"
            else:
                result["evidence_label"] = "SUPPORTED"
        else:
            result["evidence_label"] = "SUPPORTED"
    else:
        if power_at_sesoi is not None and power_at_sesoi >= 0.8:
            result["evidence_label"] = "REFUTED"
        else:
            result["evidence_label"] = "UNDERPOWERED"

    return result


def _compare_phase2b(verdict: WorkflowVerdict, phase2b: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    if not phase2b:
        return ["phase2b_results_missing"]

    def _float_close(a: Optional[float], b: Optional[float]) -> bool:
        try:
            if a is None or b is None:
                return False
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except Exception:
            return False

    if verdict.p_value is not None and "p_value" in phase2b:
        if not _float_close(verdict.p_value, phase2b.get("p_value")):
            mismatches.append("p_value_mismatch")

    if verdict.effect_size is not None and "effect_size" in phase2b:
        if not _float_close(verdict.effect_size, phase2b.get("effect_size")):
            mismatches.append("effect_size_mismatch")

    if verdict.test_used and "test_performed" in phase2b:
        v_test = _normalize_test_name(verdict.test_used)
        p_test = _normalize_test_name(phase2b.get("test_performed"))
        if v_test and p_test and v_test != p_test:
            # Only flag as mismatch if they're from different families
            if not _test_names_same_family(v_test, p_test):
                mismatches.append("test_used_mismatch")

    sample_sizes = None
    if isinstance(phase2b.get("sample_sizes"), dict):
        sample_sizes = phase2b.get("sample_sizes")

    if isinstance(getattr(verdict, "sample_sizes", None), dict) and sample_sizes:
        for key, value in sample_sizes.items():
            if key in verdict.sample_sizes and verdict.sample_sizes[key] != value:
                mismatches.append("sample_sizes_mismatch")
                break

    return mismatches


def _derive_warnings(verdict: WorkflowVerdict, phase2b: Optional[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []

    def _pick_value(key: str):
        if phase2b and key in phase2b:
            return phase2b.get(key)
        return getattr(verdict, key, None)

    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    def _is_finite(value: Any) -> bool:
        try:
            v = float(value)
            return v == v and v not in [float("inf"), float("-inf")]
        except Exception:
            return False

    p_value = _pick_value("p_value")
    if p_value is not None:
        if not _is_number(p_value):
            warnings.append("p_value_not_numeric")
        else:
            pv = float(p_value)
            if pv < 0 or pv > 1:
                warnings.append("p_value_out_of_range")
    else:
        warnings.append("p_value_missing")

    effect_size = _pick_value("effect_size")
    if effect_size is not None:
        if not _is_number(effect_size) or not _is_finite(effect_size):
            warnings.append("effect_size_not_finite")
    else:
        warnings.append("effect_size_missing")

    effect_size_type = None
    if phase2b and isinstance(phase2b.get("effect_size_type"), str):
        effect_size_type = phase2b.get("effect_size_type").lower().replace("-", "_")
    if effect_size_type in {"rank_biserial", "rank_biserial_correlation"} and _is_number(effect_size):
        if abs(float(effect_size)) > 1.0:
            warnings.append("rank_biserial_out_of_range")

    # Sign consistency: effect_size sign should match mean difference direction
    # Skip for rank_biserial — MWU sign convention differs from subtraction order
    if phase2b and _is_number(effect_size) and effect_size_type in {"cohens_d"}:
        groups = _groups_from_phase2b(phase2b)
        if groups and len(groups) >= 2:
            mean_diff = _compute_group_mean_diff(phase2b, groups)
            if mean_diff is not None and abs(mean_diff) > 0.01:  # non-trivial difference
                effect_sign = 1 if float(effect_size) > 0 else -1 if float(effect_size) < 0 else 0
                diff_sign = 1 if mean_diff > 0 else -1 if mean_diff < 0 else 0
                if effect_sign != 0 and diff_sign != 0 and effect_sign != diff_sign:
                    warnings.append("effect_sign_inconsistent_with_mean_diff")

    # Verdict sanity vs p-value (soft warning)
    verdict_value = (verdict.verdict or "").upper()
    if verdict_value in {"YES", "NO"} and _is_number(p_value):
        pv = float(p_value)
        if verdict_value == "YES" and pv >= 0.05:
            warnings.append("verdict_yes_but_p_not_significant")
        if verdict_value == "NO" and pv < 0.05:
            direction_intent = getattr(verdict, "direction_intent", None)
            observed_direction = _infer_observed_direction(phase2b, getattr(verdict, "direction_groups", None))
            conflict = True
            if direction_intent in {"positive", "negative"} and observed_direction is not None:
                if direction_intent == "positive" and observed_direction < 0:
                    conflict = False
                if direction_intent == "negative" and observed_direction > 0:
                    conflict = False
            if conflict:
                warnings.append("verdict_no_but_p_significant")

    if verdict.test_used is None:
        warnings.append("test_used_missing")
    if verdict.sample_sizes is None:
        warnings.append("sample_sizes_missing")
    if phase2b is not None:
        n_total = _safe_float(phase2b.get("n_total"))
        if n_total is None:
            warnings.append("n_total_missing")

    return warnings


def _check_sample_utilization(
    output_dir: Path,
    phase2b: Optional[Dict[str, Any]],
    verdict: "WorkflowVerdict",
    threshold: float = 0.8,
) -> None:
    """Flag when Phase 2B uses far fewer patients than Phase 2A segmented.

    Compares n_total from statistical_results.json against the number of
    unique patients segmented in Phase 2A.  A ratio below *threshold*
    triggers a ``low_sample_utilization`` warning, which helps catch coding
    agents that artificially limit sample size (e.g. ``max_patients = 20``).
    """
    if not phase2b:
        return
    n_total = phase2b.get("n_total")
    if not n_total or not isinstance(n_total, (int, float)):
        return

    unique_patients = _count_phase2a_unique_patients(output_dir)
    verdict.phase2a_unique_patients = unique_patients
    if unique_patients is None or unique_patients <= 0:
        return

    coverage = float(n_total) / float(unique_patients)
    verdict.phase2b_sample_coverage = coverage
    if coverage < threshold:
        verdict.warnings.append("low_sample_utilization")
    if coverage < 0.5:
        verdict.warnings.append("critical_sample_utilization")


def _load_workflow_config(output_dir: Path) -> Dict[str, Any]:
    config_path = output_dir / "workflow_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _count_phase2a_unique_patients(output_dir: Path) -> Optional[int]:
    req_path = output_dir / "phase2a_imaging_analysis" / "workspace" / "segmentation_request.json"
    if not req_path.exists():
        return None
    try:
        with open(req_path) as f:
            request = json.load(f)
    except Exception:
        return None
    identifiers = request.get("identifiers")
    if not isinstance(identifiers, list):
        return None
    patients = set()
    for token in identifiers:
        if not isinstance(token, str):
            continue
        parts = token.split(":")
        if len(parts) >= 3 and parts[1].strip():
            patients.add(parts[1].strip())
    return len(patients) if patients else None


def _detect_literal_pvalue_assignment(output_dir: Path) -> bool:
    files_to_scan = _get_phase2b_active_code_files(output_dir)
    if not files_to_scan:
        return False
    literal_patterns = [
        re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*p[_a-zA-Z0-9]*\s*=\s*0\.0*5(?:0+)?\b"),
        re.compile(r"\bp[_ ]?value\s*=\s*0\.0*5(?:0+)?\b"),
    ]
    for py_file in files_to_scan:
        try:
            text = py_file.read_text(errors="ignore")
        except Exception:
            continue
        for pattern in literal_patterns:
            if pattern.search(text):
                return True
    return False


def _check_phase2b_data_curation(
    output_dir: Path,
    phase2b: Optional[Dict[str, Any]],
    verdict: "WorkflowVerdict",
) -> None:
    if not phase2b:
        return

    config = _load_workflow_config(output_dir)
    plan = config.get("plan", {}) if isinstance(config, dict) else {}
    groups = [str(g).strip() for g in (plan.get("groups") or []) if str(g).strip()]
    cohort_mode = str(plan.get("cohort_mode", "")).strip().lower()
    grouping_field = str(plan.get("grouping_field") or "").strip().lower()
    group_spec = plan.get("group_spec") if isinstance(plan.get("group_spec"), dict) else {}
    group_spec_type = str(group_spec.get("type", "")).strip().lower()
    all_groups_selected = cohort_mode == "all" or (len(groups) == 1 and groups[0].lower() == "all")

    sample_sizes = phase2b.get("sample_sizes") if isinstance(phase2b.get("sample_sizes"), dict) else {}
    sample_groups = [str(g).strip() for g in sample_sizes.keys() if str(g).strip()]
    n_total = _safe_float(phase2b.get("n_total"))
    sum_sizes = sum(v for v in sample_sizes.values() if isinstance(v, (int, float))) if sample_sizes else None
    if n_total is not None and sum_sizes is not None:
        verdict.phase2b_n_total_sum_mismatch = int(n_total) != int(sum_sizes)
        if verdict.phase2b_n_total_sum_mismatch:
            verdict.warnings.append("n_total_sample_sizes_mismatch")
    else:
        verdict.phase2b_n_total_sum_mismatch = None

    # Only enforce plan-group conformance for direct dataset group selections.
    dataset_group_mode = (
        bool(groups)
        and not all_groups_selected
        and not grouping_field
        and group_spec_type in {"", "dataset"}
    )
    if dataset_group_mode and sample_groups:
        plan_set = set(groups)
        sample_set = set(sample_groups)
        subset_violation = not sample_set.issubset(plan_set)
        missing_groups = sorted([g for g in groups if g not in sample_set])
        verdict.phase2b_group_subset_violation = subset_violation
        verdict.phase2b_missing_planned_groups = missing_groups
        if subset_violation:
            verdict.warnings.append("sample_group_not_in_plan")
        if missing_groups:
            verdict.warnings.append("planned_group_missing_in_samples")
    else:
        verdict.phase2b_group_subset_violation = None
        verdict.phase2b_missing_planned_groups = []

    p_value = _safe_float(phase2b.get("p_value"))
    verdict.phase2b_boundary_pvalue_005 = (
        p_value is not None and abs(p_value - 0.05) <= 1e-12
    )
    if verdict.phase2b_boundary_pvalue_005:
        verdict.warnings.append("p_value_boundary_0_05")

    verdict.phase2b_literal_pvalue_assignment = _detect_literal_pvalue_assignment(output_dir)
    if verdict.phase2b_literal_pvalue_assignment:
        verdict.warnings.append("literal_p_value_assignment_detected")


def extract_verdict_from_json_block(content: str) -> Dict[str, Any]:
    """
    Extract JSON verdict block from Phase 3 output.

    Looks for structured JSON blocks that may have been added to the agenda.
    """
    patterns = [
        r'```json\s*(\{[^`]*"verdict"[^`]*\})\s*```',
        r'FINAL VERDICT.*?(\{[^}]*"verdict"[^}]*\})',
        r'"verdict"\s*:\s*"(YES|NO|INCONCLUSIVE)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                if pattern.endswith('"'):
                    # Simple verdict extraction
                    return {"verdict": match.group(1).upper()}
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return {}


def extract_verdict_from_text(content: str) -> Dict[str, Any]:
    """
    Fallback regex extraction for verdict from free-form text.
    """
    result = {}

    # Try to find YES/NO/INCONCLUSIVE
    verdict_patterns = [
        (r'\*\*(?:Conclusion|Verdict|Result)\*\*:?\s*\*?\*?(YES|NO|INCONCLUSIVE)', 'verdict'),
        (r'Hypothesis\s+(?:Status|Conclusion):?\s*(YES|NO|INCONCLUSIVE)', 'verdict'),
        (r'(?:is|are)\s+(supported|rejected|not supported|inconclusive)', 'verdict_text'),
        (r'"verdict":\s*"(YES|NO|INCONCLUSIVE)"', 'verdict'),
    ]

    for pattern, key in verdict_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).upper()
            if key == 'verdict_text':
                # Map text to verdict
                if value in ['SUPPORTED']:
                    value = 'YES'
                elif value in ['REJECTED', 'NOT SUPPORTED']:
                    value = 'NO'
                else:
                    value = 'INCONCLUSIVE'
            result['verdict'] = value
            break

    # Extract p-value
    p_patterns = [
        r'p\s*[<>=]\s*([\d.e-]+)',
        r'p-value[:\s]*([\d.e-]+)',
        r'"p_value":\s*([\d.e-]+)',
    ]
    for pattern in p_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                result['p_value'] = float(match.group(1))
            except ValueError:
                pass
            break

    # Extract effect size
    d_patterns = [
        r"Cohen'?s?\s*d\s*[=:]\s*([-]?[\d.]+)",
        r'"effect_size":\s*([-]?[\d.]+)',
        r'effect\s*size[:\s]*([-]?[\d.]+)',
        r'd\s*=\s*([-]?[\d.]+)',
    ]
    for pattern in d_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                result['effect_size'] = float(match.group(1))
            except ValueError:
                pass
            break

    return result


def extract_workflow_verdict(
    output_dir: Path,
    hypothesis_id: str,
    run_id: str,
    ground_truth: Optional[Dict] = None,
    hypothesis: Optional[Dict] = None,
    dataset_registry: Optional[Dict] = None,
    sesoi_override: Optional[Any] = None,
) -> WorkflowVerdict:
    """
    Extract verdict from workflow output directory.

    Uses a three-step cascade: config → json_block → regex.

    Args:
        output_dir: Path to workflow output directory
        hypothesis_id: ID of the hypothesis being tested
        run_id: ID of this run
        ground_truth: Optional ground truth dict with expected_conclusion
        hypothesis: Optional hypothesis config for evidence label computation
        dataset_registry: Optional dataset registry for SESOI profiles
        sesoi_override: Optional SESOI override (profile name or numeric value)

    Returns:
        WorkflowVerdict with extracted information
    """
    output_dir = Path(output_dir)
    phase3_dir = output_dir / "phase3_interpretation"

    # Check if workflow completed
    discussion_json = phase3_dir / "discussion.json"
    discussion_md = phase3_dir / "discussion.md"

    phase3_available = discussion_json.exists() or discussion_md.exists()
    parse_error: Optional[str] = None
    completed_all_phases = phase3_available

    # Read content
    try:
        final_content = ""

        # Try discussion.json first (has structured messages)
        if discussion_json.exists():
            with open(discussion_json) as f:
                discussion = json.load(f)

            # Get final PI message (usually last agent message)
            for msg in reversed(discussion):
                if msg.get("agent") == "Principal Investigator":
                    final_content = msg.get("message", "")
                    break

        # Fallback to discussion.md
        if not final_content and discussion_md.exists():
            with open(discussion_md) as f:
                final_content = f.read()

        if not final_content:
            parse_error = "No content found in Phase 3 outputs"

    except Exception as e:
        parse_error = str(e)
        final_content = ""

    if not phase3_available and not parse_error:
        parse_error = "Phase 3 outputs not found"

    # Try extraction methods in order of preference
    verdict_data = {}
    parse_method = "failed"

    # 1. Prefer the aligned verdict from workflow_config.json (full pipeline output)
    config_path = output_dir / "workflow_config.json"
    wf_config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                wf_config = json.load(f)
            saved = wf_config.get("verdict")
            if isinstance(saved, dict) and saved.get("verdict"):
                verdict_data = saved
                parse_method = "config"
        except Exception:
            pass

    # 2. Try JSON block extraction from Phase 3 text
    if not verdict_data.get("verdict"):
        verdict_data = extract_verdict_from_json_block(final_content)
        if verdict_data.get("verdict"):
            parse_method = "json_block"

    # 3. Fallback to regex
    if not verdict_data.get("verdict") or verdict_data.get("verdict") == "PARSE_ERROR":
        verdict_data = extract_verdict_from_text(final_content)
        if verdict_data.get("verdict"):
            parse_method = "regex"

    # Build verdict object
    verdict = WorkflowVerdict(
        hypothesis_id=hypothesis_id,
        run_id=run_id,
        output_dir=str(output_dir),
        verdict=verdict_data.get("verdict", "PARSE_ERROR"),
        p_value=verdict_data.get("p_value"),
        effect_size=verdict_data.get("effect_size"),
        test_used=verdict_data.get("test_used"),
        sample_sizes=verdict_data.get("sample_sizes"),
        n_total=verdict_data.get("n_total"),
        confidence=verdict_data.get("confidence"),
        reasoning=verdict_data.get("reasoning"),
        evidence_label_reported=verdict_data.get("evidence_label"),
        completed_all_phases=completed_all_phases,
        parse_method=parse_method,
        parse_error=verdict_data.get("error") or parse_error
    )

    workflow_audit = wf_config.get("workflow_audit") if isinstance(wf_config, dict) else None
    if not isinstance(workflow_audit, dict):
        workflow_audit = _load_workflow_audit(output_dir)
    verdict.failure_code = workflow_audit.get("failure_code") if isinstance(workflow_audit, dict) else None
    verdict.failure_stage = workflow_audit.get("failure_stage") if isinstance(workflow_audit, dict) else None

    if isinstance(workflow_audit, dict):
        phase_status = workflow_audit.get("phase_status")
        if isinstance(phase_status, dict):
            required_stages = ("phase1", "phase2a", "phase2b", "phase3")
            completed_all_phases = all(
                isinstance(phase_status.get(stage), dict)
                and phase_status.get(stage, {}).get("status") == "passed"
                for stage in required_stages
            )
            verdict.completed_all_phases = completed_all_phases

    phase1_feas = _load_phase1_feasibility(output_dir)
    verdict.phase1_feasibility_status = phase1_feas.get("status")
    verdict.phase1_invalid_subtype = phase1_feas.get("invalid_subtype")
    if verdict.phase1_feasibility_status == "UNTESTABLE" and not phase3_available:
        verdict.early_stop_type = "phase1_untestable"
    if not verdict.parse_error and verdict.failure_code and verdict.early_stop_type != "phase1_untestable":
        failure_message = workflow_audit.get("failure_message") if isinstance(workflow_audit, dict) else None
        verdict.parse_error = (
            f"{verdict.failure_code}: {failure_message}"
            if failure_message
            else verdict.failure_code
        )

    phase2b_results = _load_phase2b_results(output_dir)
    evidence = compute_evidence_label(
        phase2b_results,
        hypothesis or {},
        dataset_registry,
        output_dir,
        sesoi_override=sesoi_override,
    )
    verdict.evidence_label = evidence.get("evidence_label")
    verdict.evidence_label_raw = verdict.evidence_label
    verdict.power_at_sesoi = evidence.get("power_at_sesoi")
    verdict.sesoi_value = evidence.get("sesoi_value")
    verdict.sesoi_profile = evidence.get("sesoi_profile")
    verdict.test_family = evidence.get("test_family")
    verdict.direction_intent = evidence.get("direction_intent")
    verdict.direction_groups = evidence.get("direction_groups")
    verdict.n_total = evidence.get("n_total") or verdict.n_total

    mismatches = _compare_phase2b(verdict, phase2b_results)
    if mismatches:
        verdict.phase2b_consistent = False
        verdict.phase2b_mismatches = mismatches
    else:
        verdict.phase2b_consistent = True
    verdict.warnings = _derive_warnings(verdict, phase2b_results) + verdict.phase2b_mismatches
    verdict.warnings.extend(evidence.get("warnings", []))

    # Sample utilization check: flag when Phase 2B uses far fewer patients than Phase 2A segmented
    _check_sample_utilization(output_dir, phase2b_results, verdict)
    _check_phase2b_data_curation(output_dir, phase2b_results, verdict)

    invalid_reasons = list(evidence.get("invalid_reasons", []))
    synthetic_scan = _detect_synthetic_phase2b_usage(output_dir)
    synthetic_hard_detected = bool(synthetic_scan and synthetic_scan.get("count", 0) > 0)
    synthetic_soft_detected = bool(synthetic_scan and synthetic_scan.get("soft_count", 0) > 0)
    if synthetic_hard_detected:
        invalid_reasons.append("synthetic_data_detected")
    if synthetic_soft_detected:
        verdict.warnings.append("random_usage_detected_phase2b_code")

    # Plan vs execution test-name check (same family -> warning only)
    plan_test = _load_plan_statistical_test(output_dir)
    phase_test = phase2b_results.get("test_performed") if phase2b_results else None
    if plan_test and phase_test:
        plan_norm = _normalize_test_name(plan_test)
        phase_norm = _normalize_test_name(str(phase_test))
        if plan_norm != phase_norm:
            plan_family = _infer_test_family_from_name(plan_norm)
            phase_family = _infer_test_family_from_name(phase_norm)
            if plan_family and plan_family == phase_family:
                verdict.warnings.append("test_name_differs_from_plan")

    # Internal consistency: Phase 3 verdict vs numeric evidence
    if verdict.verdict in {"YES", "NO"} and verdict.p_value is not None:
        pv = _safe_float(verdict.p_value)
        if pv is not None and verdict.verdict == "YES" and pv >= 0.05:
            invalid_reasons.append("verdict_conflicts_with_p")
        if pv is not None and verdict.verdict == "NO" and pv < 0.05:
            observed_direction = _infer_observed_direction(phase2b_results, verdict.direction_groups)
            direction_intent = verdict.direction_intent
            conflict = True
            if direction_intent in {"positive", "negative"} and observed_direction is not None:
                if direction_intent == "positive" and observed_direction < 0:
                    conflict = False
                if direction_intent == "negative" and observed_direction > 0:
                    conflict = False
            if conflict:
                invalid_reasons.append("verdict_conflicts_with_p")

    if invalid_reasons:
        verdict.evidence_label = "INVALID"
        verdict.warnings.extend(invalid_reasons)
        if synthetic_hard_detected:
            verdict.warnings.append("synthetic_data_detected_phase2b_code")

    # Track whether agent verdict aligns with evidence label (advisory only)
    _evidence_verdict_map = {
        "SUPPORTED": "YES",
        "REFUTED": "NO",
        "UNDERPOWERED": "INCONCLUSIVE",
        "INVALID": "INCONCLUSIVE",
    }
    expected_from_evidence = _evidence_verdict_map.get(str(verdict.evidence_label or "").upper())
    if expected_from_evidence and verdict.verdict != expected_from_evidence:
        verdict.warnings.append("verdict_evidence_mismatch")

    # Compare to ground truth if provided
    if ground_truth:
        expected = ground_truth.get("expected_conclusion")
        verdict.expected_conclusion = expected
        if expected and verdict.verdict not in ["PARSE_ERROR"]:
            verdict.is_correct = (verdict.verdict == expected)
        expected_label = ground_truth.get("expected_label")
        verdict.expected_label = expected_label
        verdict.expected_invalid_subtype = ground_truth.get("expected_invalid_subtype")
        if expected_label and verdict.evidence_label:
            verdict.is_label_correct = (verdict.evidence_label == expected_label)
        verdict.has_dataset_limitation = bool(ground_truth.get("dataset_limitation"))
        if (
            synthetic_hard_detected
            and (str(expected_label or "").upper() == "INVALID")
            and verdict.phase1_feasibility_status != "UNTESTABLE"
        ):
            verdict.warnings.append("l0_feasibility_miss_recovered_by_integrity_guard")

    if hypothesis:
        complexity = hypothesis.get("complexity", {})
        control = hypothesis.get("control", {})
        verdict.complexity_tier = complexity.get("tier")
        verdict.control_type = control.get("type")
        verdict.control_category = control.get("category")

    # Load coding trial counts from phase execution metadata
    trial_counts = _load_coding_trials(output_dir)
    verdict.coding_trials_phase2a = trial_counts.get("phase2a")
    verdict.coding_trials_phase2b = trial_counts.get("phase2b")
    t2a = verdict.coding_trials_phase2a or 0
    t2b = verdict.coding_trials_phase2b or 0
    if t2a > 0 or t2b > 0:
        verdict.coding_trials_total = t2a + t2b

    return verdict


def evaluate_batch(
    results_dir: Path,
    hypothesis_bank: List[Dict],
    dataset_registry: Optional[Dict[str, Any]] = None,
    output_file: Optional[Path] = None,
    sesoi_override: Optional[Any] = None,
) -> List[WorkflowVerdict]:
    """
    Evaluate multiple workflow runs against hypothesis bank.

    Args:
        results_dir: Directory containing workflow outputs
        hypothesis_bank: List of hypothesis definitions
        dataset_registry: Optional dataset registry for SESOI profiles
        output_file: Optional path to save results JSON
        sesoi_override: Optional SESOI override (profile name or numeric value)

    Returns:
        List of WorkflowVerdict objects
    """
    verdicts = []
    results_dir = Path(results_dir)

    looks_like_single_run = any(
        (results_dir / phase_dir).exists()
        for phase_dir in (
            "phase1_hypothesis_formulation",
            "phase2a_imaging_analysis",
            "phase2b_statistical_analysis",
            "phase3_interpretation",
        )
    )

    for hypothesis in hypothesis_bank:
        hyp_id = hypothesis["id"]
        ground_truth = hypothesis.get("ground_truth", {})

        # Find all runs for this hypothesis
        hyp_dir = results_dir / hyp_id
        if hyp_dir.exists():
            run_dirs = sorted([d for d in hyp_dir.iterdir() if d.is_dir()])
            if not run_dirs:
                # Single run without run_* suffix
                run_dirs = [hyp_dir]
        else:
            # Support direct single-run evaluation only when results_dir itself
            # is a workflow run and the bank contains a single hypothesis.
            if looks_like_single_run and len(hypothesis_bank) == 1:
                run_dirs = [results_dir]
            else:
                continue

        for run_dir in run_dirs:
            if run_dir.is_dir():
                run_id = run_dir.name if run_dir != results_dir else "run_000"
                verdict = extract_workflow_verdict(
                    run_dir, hyp_id, run_id, ground_truth,
                    hypothesis=hypothesis,
                    dataset_registry=dataset_registry,
                    sesoi_override=sesoi_override,
                )
                verdicts.append(verdict)

    # Save results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump([asdict(v) for v in verdicts], f, indent=2)

    return verdicts


def compute_metrics(verdicts: List[WorkflowVerdict]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from verdicts.

    Returns accuracy, per-class metrics, and consistency information.
    """
    valid = [v for v in verdicts if v.expected_conclusion and v.verdict != "PARSE_ERROR" and not v.has_dataset_limitation]
    label_valid = [v for v in verdicts if v.expected_label and v.evidence_label and not v.has_dataset_limitation]
    raw_label_valid = [v for v in verdicts if v.expected_label and v.evidence_label_raw and not v.has_dataset_limitation]
    valid_non_l0 = [v for v in valid if (v.complexity_tier or "").upper() != "L0"]
    label_valid_non_l0 = [v for v in label_valid if (v.complexity_tier or "").upper() != "L0"]
    raw_label_valid_non_l0 = [v for v in raw_label_valid if (v.complexity_tier or "").upper() != "L0"]

    if not valid and not label_valid and not raw_label_valid:
        return {"error": "No valid verdicts with ground truth"}

    correct = sum(1 for v in valid if v.is_correct)
    total = len(valid)
    label_correct = sum(1 for v in label_valid if v.is_label_correct)
    label_total = len(label_valid)
    correct_non_l0 = sum(1 for v in valid_non_l0 if v.is_correct)
    total_non_l0 = len(valid_non_l0)
    label_correct_non_l0 = sum(1 for v in label_valid_non_l0 if v.is_label_correct)
    label_total_non_l0 = len(label_valid_non_l0)
    raw_label_correct = sum(
        1 for v in raw_label_valid
        if str(v.evidence_label_raw or "").upper() == str(v.expected_label or "").upper()
    )
    raw_label_total = len(raw_label_valid)
    raw_label_correct_non_l0 = sum(
        1 for v in raw_label_valid_non_l0
        if str(v.evidence_label_raw or "").upper() == str(v.expected_label or "").upper()
    )
    raw_label_total_non_l0 = len(raw_label_valid_non_l0)

    by_expected: Dict[str, Any] = {}
    for v in valid:
        exp = v.expected_conclusion
        if exp not in by_expected:
            by_expected[exp] = {"total": 0, "correct": 0, "predicted": []}
        by_expected[exp]["total"] += 1
        by_expected[exp]["predicted"].append(v.verdict)
        if v.is_correct:
            by_expected[exp]["correct"] += 1
    for exp in by_expected:
        by_expected[exp]["accuracy"] = (
            by_expected[exp]["correct"] / by_expected[exp]["total"]
            if by_expected[exp]["total"] > 0 else 0
        )

    by_expected_label: Dict[str, Any] = {}
    for v in label_valid:
        exp = v.expected_label
        if exp not in by_expected_label:
            by_expected_label[exp] = {"total": 0, "correct": 0, "predicted": []}
        by_expected_label[exp]["total"] += 1
        by_expected_label[exp]["predicted"].append(v.evidence_label)
        if v.is_label_correct:
            by_expected_label[exp]["correct"] += 1
    for exp in by_expected_label:
        by_expected_label[exp]["accuracy"] = (
            by_expected_label[exp]["correct"] / by_expected_label[exp]["total"]
            if by_expected_label[exp]["total"] > 0 else 0
        )

    by_expected_label_raw: Dict[str, Any] = {}
    for v in raw_label_valid:
        exp = v.expected_label
        if exp not in by_expected_label_raw:
            by_expected_label_raw[exp] = {"total": 0, "correct": 0, "predicted": []}
        by_expected_label_raw[exp]["total"] += 1
        by_expected_label_raw[exp]["predicted"].append(v.evidence_label_raw)
        if str(v.evidence_label_raw or "").upper() == str(v.expected_label or "").upper():
            by_expected_label_raw[exp]["correct"] += 1
    for exp in by_expected_label_raw:
        by_expected_label_raw[exp]["accuracy"] = (
            by_expected_label_raw[exp]["correct"] / by_expected_label_raw[exp]["total"]
            if by_expected_label_raw[exp]["total"] > 0 else 0
        )

    # Tier-stratified and control-stratified metrics
    by_tier: Dict[str, Dict[str, Any]] = {}
    by_control_type: Dict[str, Dict[str, Any]] = {}
    for v in verdicts:
        tier = v.complexity_tier or "UNKNOWN"
        control = v.control_type or "unknown"
        if tier not in by_tier:
            by_tier[tier] = {
                "total_runs": 0, "label_valid": 0, "label_correct": 0,
                "raw_label_valid": 0, "raw_label_correct": 0,
                "verdict_valid": 0, "verdict_correct": 0,
            }
        if control not in by_control_type:
            by_control_type[control] = {
                "total_runs": 0, "label_valid": 0, "label_correct": 0,
                "raw_label_valid": 0, "raw_label_correct": 0,
                "verdict_valid": 0, "verdict_correct": 0,
            }

        by_tier[tier]["total_runs"] += 1
        by_control_type[control]["total_runs"] += 1

        if v.expected_label and v.evidence_label and not v.has_dataset_limitation:
            by_tier[tier]["label_valid"] += 1
            by_control_type[control]["label_valid"] += 1
            if v.is_label_correct:
                by_tier[tier]["label_correct"] += 1
                by_control_type[control]["label_correct"] += 1

        if v.expected_label and v.evidence_label_raw and not v.has_dataset_limitation:
            by_tier[tier]["raw_label_valid"] += 1
            by_control_type[control]["raw_label_valid"] += 1
            if str(v.evidence_label_raw or "").upper() == str(v.expected_label or "").upper():
                by_tier[tier]["raw_label_correct"] += 1
                by_control_type[control]["raw_label_correct"] += 1

        if v.expected_conclusion and v.verdict != "PARSE_ERROR" and not v.has_dataset_limitation:
            by_tier[tier]["verdict_valid"] += 1
            by_control_type[control]["verdict_valid"] += 1
            if v.is_correct:
                by_tier[tier]["verdict_correct"] += 1
                by_control_type[control]["verdict_correct"] += 1

    for data in by_tier.values():
        data["evidence_label_accuracy"] = (
            data["label_correct"] / data["label_valid"] if data["label_valid"] > 0 else None
        )
        data["evidence_label_raw_accuracy"] = (
            data["raw_label_correct"] / data["raw_label_valid"] if data["raw_label_valid"] > 0 else None
        )
        data["verdict_accuracy"] = (
            data["verdict_correct"] / data["verdict_valid"] if data["verdict_valid"] > 0 else None
        )
    for data in by_control_type.values():
        data["evidence_label_accuracy"] = (
            data["label_correct"] / data["label_valid"] if data["label_valid"] > 0 else None
        )
        data["evidence_label_raw_accuracy"] = (
            data["raw_label_correct"] / data["raw_label_valid"] if data["raw_label_valid"] > 0 else None
        )
        data["verdict_accuracy"] = (
            data["verdict_correct"] / data["verdict_valid"] if data["verdict_valid"] > 0 else None
        )

    # Consistency check (for repeated runs of same hypothesis)
    consistency = {}
    by_hypothesis_rows: Dict[str, List[WorkflowVerdict]] = {}
    by_hypothesis_verdicts: Dict[str, List[str]] = {}
    for v in verdicts:
        if v.hypothesis_id not in by_hypothesis_rows:
            by_hypothesis_rows[v.hypothesis_id] = []
            by_hypothesis_verdicts[v.hypothesis_id] = []
        by_hypothesis_rows[v.hypothesis_id].append(v)
        by_hypothesis_verdicts[v.hypothesis_id].append(v.verdict)
    for hyp_id, vlist in by_hypothesis_verdicts.items():
        if len(vlist) > 1:
            most_common = max(set(vlist), key=vlist.count)
            consistency[hyp_id] = {
                "verdicts": vlist,
                "consistency_rate": vlist.count(most_common) / len(vlist),
                "most_common": most_common,
            }

    # Hypothesis-level aggregation across repeated runs (majority vote)
    majority_vote = {
        "verdict": {"total": 0, "correct": 0, "ties": 0},
        "evidence_label": {"total": 0, "correct": 0, "ties": 0},
    }
    majority_vote_e2e = {
        "verdict": {"total": 0, "correct": 0, "ties": 0, "execution_failure_majority": 0},
        "evidence_label": {"total": 0, "correct": 0, "ties": 0, "execution_failure_majority": 0},
    }
    for hyp_id, hyp_rows in by_hypothesis_rows.items():

        verdict_rows = [
            v for v in hyp_rows
            if v.expected_conclusion and v.verdict != "PARSE_ERROR" and not v.has_dataset_limitation
        ]
        if verdict_rows:
            counts = Counter(v.verdict for v in verdict_rows)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            majority_vote["verdict"]["total"] += 1
            if len(top_vals) > 1:
                majority_vote["verdict"]["ties"] += 1
            else:
                expected = verdict_rows[0].expected_conclusion
                if top_vals[0] == expected:
                    majority_vote["verdict"]["correct"] += 1

        label_rows = [
            v for v in hyp_rows
            if v.expected_label and v.evidence_label and not v.has_dataset_limitation
        ]
        if label_rows:
            counts = Counter(str(v.evidence_label) for v in label_rows)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            majority_vote["evidence_label"]["total"] += 1
            if len(top_vals) > 1:
                majority_vote["evidence_label"]["ties"] += 1
            else:
                expected = str(label_rows[0].expected_label)
                if top_vals[0] == expected:
                    majority_vote["evidence_label"]["correct"] += 1

    for key in ("verdict", "evidence_label"):
        total_h = majority_vote[key]["total"]
        ties = majority_vote[key]["ties"]
        non_tied = total_h - ties
        majority_vote[key]["non_tied_total"] = non_tied
        majority_vote[key]["accuracy"] = (
            majority_vote[key]["correct"] / total_h if total_h > 0 else None
        )
        majority_vote[key]["accuracy_non_tied"] = (
            majority_vote[key]["correct"] / non_tied if non_tied > 0 else None
        )

    # Strict end-to-end majority vote:
    # include all attempts and map runtime/parse failures to an explicit failure token.
    exec_fail_token = "__EXECUTION_FAILURE__"
    for hyp_id, hyp_rows in by_hypothesis_rows.items():
        verdict_values: List[str] = []
        verdict_expected: Optional[str] = None
        for v in hyp_rows:
            if not v.expected_conclusion or v.has_dataset_limitation:
                continue
            if verdict_expected is None:
                verdict_expected = v.expected_conclusion
            is_runtime_failure = bool(v.parse_error and v.early_stop_type != "phase1_untestable")
            value = (
                exec_fail_token
                if (is_runtime_failure or v.verdict == "PARSE_ERROR" or not v.verdict)
                else v.verdict
            )
            verdict_values.append(str(value))
        if verdict_values and verdict_expected is not None:
            majority_vote_e2e["verdict"]["total"] += 1
            counts = Counter(verdict_values)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            if len(top_vals) > 1:
                majority_vote_e2e["verdict"]["ties"] += 1
            else:
                top = top_vals[0]
                if top == exec_fail_token:
                    majority_vote_e2e["verdict"]["execution_failure_majority"] += 1
                if top == verdict_expected:
                    majority_vote_e2e["verdict"]["correct"] += 1

        label_values: List[str] = []
        label_expected: Optional[str] = None
        for v in hyp_rows:
            if not v.expected_label or v.has_dataset_limitation:
                continue
            if label_expected is None:
                label_expected = str(v.expected_label)
            is_runtime_failure = bool(v.parse_error and v.early_stop_type != "phase1_untestable")
            value = (
                exec_fail_token
                if (is_runtime_failure or not v.evidence_label)
                else str(v.evidence_label)
            )
            label_values.append(str(value))
        if label_values and label_expected is not None:
            majority_vote_e2e["evidence_label"]["total"] += 1
            counts = Counter(label_values)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            if len(top_vals) > 1:
                majority_vote_e2e["evidence_label"]["ties"] += 1
            else:
                top = top_vals[0]
                if top == exec_fail_token:
                    majority_vote_e2e["evidence_label"]["execution_failure_majority"] += 1
                if top == label_expected:
                    majority_vote_e2e["evidence_label"]["correct"] += 1

    for key in ("verdict", "evidence_label"):
        total_h = majority_vote_e2e[key]["total"]
        ties = majority_vote_e2e[key]["ties"]
        non_tied = total_h - ties
        majority_vote_e2e[key]["non_tied_total"] = non_tied
        majority_vote_e2e[key]["accuracy"] = (
            majority_vote_e2e[key]["correct"] / total_h if total_h > 0 else None
        )
        majority_vote_e2e[key]["accuracy_non_tied"] = (
            majority_vote_e2e[key]["correct"] / non_tied if non_tied > 0 else None
        )

    # Hypothesis-level aggregation restricted to non-L0 hypotheses
    majority_vote_non_l0 = {
        "verdict": {"total": 0, "correct": 0, "ties": 0},
        "evidence_label": {"total": 0, "correct": 0, "ties": 0},
    }
    majority_vote_e2e_non_l0 = {
        "verdict": {"total": 0, "correct": 0, "ties": 0, "execution_failure_majority": 0},
        "evidence_label": {"total": 0, "correct": 0, "ties": 0, "execution_failure_majority": 0},
    }
    for hyp_id, hyp_rows in by_hypothesis_rows.items():
        non_l0_rows = [v for v in hyp_rows if (v.complexity_tier or "").upper() != "L0"]
        if not non_l0_rows:
            continue

        verdict_rows = [
            v for v in non_l0_rows
            if v.expected_conclusion and v.verdict != "PARSE_ERROR" and not v.has_dataset_limitation
        ]
        if verdict_rows:
            counts = Counter(v.verdict for v in verdict_rows)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            majority_vote_non_l0["verdict"]["total"] += 1
            if len(top_vals) > 1:
                majority_vote_non_l0["verdict"]["ties"] += 1
            else:
                expected = verdict_rows[0].expected_conclusion
                if top_vals[0] == expected:
                    majority_vote_non_l0["verdict"]["correct"] += 1

        label_rows = [
            v for v in non_l0_rows
            if v.expected_label and v.evidence_label and not v.has_dataset_limitation
        ]
        if label_rows:
            counts = Counter(str(v.evidence_label) for v in label_rows)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            majority_vote_non_l0["evidence_label"]["total"] += 1
            if len(top_vals) > 1:
                majority_vote_non_l0["evidence_label"]["ties"] += 1
            else:
                expected = str(label_rows[0].expected_label)
                if top_vals[0] == expected:
                    majority_vote_non_l0["evidence_label"]["correct"] += 1

        verdict_values: List[str] = []
        verdict_expected: Optional[str] = None
        for v in non_l0_rows:
            if not v.expected_conclusion or v.has_dataset_limitation:
                continue
            if verdict_expected is None:
                verdict_expected = v.expected_conclusion
            is_runtime_failure = bool(v.parse_error and v.early_stop_type != "phase1_untestable")
            value = (
                exec_fail_token
                if (is_runtime_failure or v.verdict == "PARSE_ERROR" or not v.verdict)
                else v.verdict
            )
            verdict_values.append(str(value))
        if verdict_values and verdict_expected is not None:
            majority_vote_e2e_non_l0["verdict"]["total"] += 1
            counts = Counter(verdict_values)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            if len(top_vals) > 1:
                majority_vote_e2e_non_l0["verdict"]["ties"] += 1
            else:
                top = top_vals[0]
                if top == exec_fail_token:
                    majority_vote_e2e_non_l0["verdict"]["execution_failure_majority"] += 1
                if top == verdict_expected:
                    majority_vote_e2e_non_l0["verdict"]["correct"] += 1

        label_values: List[str] = []
        label_expected: Optional[str] = None
        for v in non_l0_rows:
            if not v.expected_label or v.has_dataset_limitation:
                continue
            if label_expected is None:
                label_expected = str(v.expected_label)
            is_runtime_failure = bool(v.parse_error and v.early_stop_type != "phase1_untestable")
            value = (
                exec_fail_token
                if (is_runtime_failure or not v.evidence_label)
                else str(v.evidence_label)
            )
            label_values.append(str(value))
        if label_values and label_expected is not None:
            majority_vote_e2e_non_l0["evidence_label"]["total"] += 1
            counts = Counter(label_values)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            if len(top_vals) > 1:
                majority_vote_e2e_non_l0["evidence_label"]["ties"] += 1
            else:
                top = top_vals[0]
                if top == exec_fail_token:
                    majority_vote_e2e_non_l0["evidence_label"]["execution_failure_majority"] += 1
                if top == label_expected:
                    majority_vote_e2e_non_l0["evidence_label"]["correct"] += 1

    for key in ("verdict", "evidence_label"):
        total_h = majority_vote_non_l0[key]["total"]
        ties = majority_vote_non_l0[key]["ties"]
        non_tied = total_h - ties
        majority_vote_non_l0[key]["non_tied_total"] = non_tied
        majority_vote_non_l0[key]["accuracy"] = (
            majority_vote_non_l0[key]["correct"] / total_h if total_h > 0 else None
        )
        majority_vote_non_l0[key]["accuracy_non_tied"] = (
            majority_vote_non_l0[key]["correct"] / non_tied if non_tied > 0 else None
        )

    for key in ("verdict", "evidence_label"):
        total_h = majority_vote_e2e_non_l0[key]["total"]
        ties = majority_vote_e2e_non_l0[key]["ties"]
        non_tied = total_h - ties
        majority_vote_e2e_non_l0[key]["non_tied_total"] = non_tied
        majority_vote_e2e_non_l0[key]["accuracy"] = (
            majority_vote_e2e_non_l0[key]["correct"] / total_h if total_h > 0 else None
        )
        majority_vote_e2e_non_l0[key]["accuracy_non_tied"] = (
            majority_vote_e2e_non_l0[key]["correct"] / non_tied if non_tied > 0 else None
        )

    # L0 majority feasibility (INVALID detection) for transparent L0 reporting
    l0_feasibility_majority = {"total": 0, "correct": 0, "ties": 0}
    l0_feasibility_majority_e2e = {"total": 0, "correct": 0, "ties": 0, "execution_failure_majority": 0}
    for hyp_id, hyp_rows in by_hypothesis_rows.items():
        l0_rows = [
            v for v in hyp_rows
            if (v.complexity_tier or "").upper() == "L0" and v.expected_label and not v.has_dataset_limitation
        ]
        if not l0_rows:
            continue

        label_values = [str(v.evidence_label) for v in l0_rows if v.evidence_label]
        if label_values:
            counts = Counter(label_values)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            l0_feasibility_majority["total"] += 1
            if len(top_vals) > 1:
                l0_feasibility_majority["ties"] += 1
            else:
                expected = str(l0_rows[0].expected_label)
                if top_vals[0] == expected:
                    l0_feasibility_majority["correct"] += 1

        label_values_e2e: List[str] = []
        expected_e2e: Optional[str] = None
        for v in l0_rows:
            if expected_e2e is None:
                expected_e2e = str(v.expected_label)
            is_runtime_failure = bool(v.parse_error and v.early_stop_type != "phase1_untestable")
            value = (
                exec_fail_token
                if (is_runtime_failure or not v.evidence_label)
                else str(v.evidence_label)
            )
            label_values_e2e.append(value)
        if label_values_e2e and expected_e2e is not None:
            counts = Counter(label_values_e2e)
            top_n = counts.most_common(1)[0][1]
            top_vals = sorted([k for k, c in counts.items() if c == top_n])
            l0_feasibility_majority_e2e["total"] += 1
            if len(top_vals) > 1:
                l0_feasibility_majority_e2e["ties"] += 1
            else:
                top = top_vals[0]
                if top == exec_fail_token:
                    l0_feasibility_majority_e2e["execution_failure_majority"] += 1
                if top == expected_e2e:
                    l0_feasibility_majority_e2e["correct"] += 1

    for bucket in (l0_feasibility_majority, l0_feasibility_majority_e2e):
        total_h = bucket["total"]
        ties = bucket["ties"]
        non_tied = total_h - ties
        bucket["non_tied_total"] = non_tied
        bucket["accuracy"] = bucket["correct"] / total_h if total_h > 0 else None
        bucket["accuracy_non_tied"] = bucket["correct"] / non_tied if non_tied > 0 else None

    verdict_consistency_rates = [
        data["consistency_rate"] for data in consistency.values()
        if data.get("consistency_rate") is not None
    ]
    consistency_summary = {
        "verdict_hypotheses": len(consistency),
        "verdict_mean_consistency": (
            sum(verdict_consistency_rates) / len(verdict_consistency_rates)
            if verdict_consistency_rates else None
        ),
    }

    parse_methods: Dict[str, int] = {}
    for v in verdicts:
        parse_methods[v.parse_method] = parse_methods.get(v.parse_method, 0) + 1

    failure_code_counts: Dict[str, int] = {}
    failure_stage_counts: Dict[str, int] = {}
    for v in verdicts:
        if v.failure_code:
            failure_code_counts[v.failure_code] = failure_code_counts.get(v.failure_code, 0) + 1
        if v.failure_stage:
            failure_stage_counts[v.failure_stage] = failure_stage_counts.get(v.failure_stage, 0) + 1

    phase2b_counts = {"consistent": 0, "inconsistent": 0, "unknown": 0}
    for v in verdicts:
        if v.phase2b_consistent is True:
            phase2b_counts["consistent"] += 1
        elif v.phase2b_consistent is False:
            phase2b_counts["inconsistent"] += 1
        else:
            phase2b_counts["unknown"] += 1

    warning_counts: Dict[str, int] = {}
    warning_counts_unique_runs: Dict[str, int] = {}
    for v in verdicts:
        for w in v.warnings:
            warning_counts[w] = warning_counts.get(w, 0) + 1
        for w in set(v.warnings):
            warning_counts_unique_runs[w] = warning_counts_unique_runs.get(w, 0) + 1

    verdict_mismatches = sum(1 for v in verdicts if "verdict_evidence_mismatch" in v.warnings)

    # New diagnostic rates
    overclaim_pool = [v for v in verdicts if v.verdict == "YES" and v.evidence_label is not None]
    overclaim_count = sum(1 for v in overclaim_pool if v.evidence_label != "SUPPORTED")

    hallucinated_sig_pool = [v for v in verdicts if v.verdict == "YES"]
    hallucinated_sig_count = sum(
        1 for v in hallucinated_sig_pool
        if (_safe_float(v.p_value) is None) or (_safe_float(v.p_value) >= 0.05)
    )
    synthetic_data_pool = [v for v in verdicts if "synthetic_data_detected" in set(v.warnings)]
    synthetic_data_count = len(synthetic_data_pool)
    random_usage_soft_pool = [v for v in verdicts if "random_usage_detected_phase2b_code" in set(v.warnings)]
    random_usage_soft_count = len(random_usage_soft_pool)
    l0_integrity_recovered_pool = [
        v for v in verdicts if "l0_feasibility_miss_recovered_by_integrity_guard" in set(v.warnings)
    ]
    l0_integrity_recovered_count = len(l0_integrity_recovered_pool)
    broad_hallucination_pool = verdicts
    broad_hallucination_count = sum(
        1
        for v in verdicts
        if (
            (v.verdict == "YES" and ((_safe_float(v.p_value) is None) or (_safe_float(v.p_value) >= 0.05)))
            or ("synthetic_data_detected" in set(v.warnings))
        )
    )

    predicted_refuted_pool = [v for v in label_valid if v.evidence_label == "REFUTED"]
    false_refutation_count = sum(
        1 for v in predicted_refuted_pool if (v.expected_label or "").upper() != "REFUTED"
    )

    l0_pool = [v for v in verdicts if (v.complexity_tier or "").upper() == "L0" and v.expected_label]
    l0_feasibility_correct = sum(
        1 for v in l0_pool if (v.evidence_label or "").upper() == "INVALID"
    )
    l0_non_early_pool = [v for v in l0_pool if v.early_stop_type != "phase1_untestable"]
    l0_non_early_correct = sum(
        1 for v in l0_non_early_pool if (v.evidence_label or "").upper() == "INVALID"
    )

    early_stop_pool = [v for v in verdicts if v.early_stop_type == "phase1_untestable"]
    l0_early_stop_pool = [v for v in early_stop_pool if (v.complexity_tier or "").upper() == "L0" and v.expected_label]
    l0_early_stop_correct = sum(
        1 for v in l0_early_stop_pool if (v.evidence_label or "").upper() == "INVALID"
    )
    non_l0_early_stop_pool = [v for v in early_stop_pool if (v.complexity_tier or "").upper() != "L0"]

    parse_error_rows_total = sum(1 for v in verdicts if v.parse_error)
    parse_runtime_failure_rows = sum(
        1 for v in verdicts if v.parse_error and v.early_stop_type != "phase1_untestable"
    )
    parse_error_verdict_rows = sum(1 for v in verdicts if v.verdict == "PARSE_ERROR")
    completed_all_phases_rows = sum(1 for v in verdicts if v.completed_all_phases)

    # Execution-aware end-to-end metrics (runtime failures count as wrong)
    label_e2e_pool = [
        v for v in verdicts if v.expected_label and not v.has_dataset_limitation
    ]
    label_e2e_correct = sum(
        1
        for v in label_e2e_pool
        if v.is_label_correct and not (v.parse_error and v.early_stop_type != "phase1_untestable")
    )
    verdict_e2e_pool = [
        v for v in verdicts if v.expected_conclusion and not v.has_dataset_limitation
    ]
    verdict_e2e_correct = sum(
        1
        for v in verdict_e2e_pool
        if v.is_correct and not (v.parse_error and v.early_stop_type != "phase1_untestable")
    )
    label_e2e_non_l0_pool = [v for v in label_e2e_pool if (v.complexity_tier or "").upper() != "L0"]
    label_e2e_non_l0_correct = sum(
        1
        for v in label_e2e_non_l0_pool
        if v.is_label_correct and not (v.parse_error and v.early_stop_type != "phase1_untestable")
    )
    verdict_e2e_non_l0_pool = [v for v in verdict_e2e_pool if (v.complexity_tier or "").upper() != "L0"]
    verdict_e2e_non_l0_correct = sum(
        1
        for v in verdict_e2e_non_l0_pool
        if v.is_correct and not (v.parse_error and v.early_stop_type != "phase1_untestable")
    )

    # Completed-run metrics (both reasoning + execution succeeded)
    label_completed_pool = [
        v
        for v in verdicts
        if v.completed_all_phases and v.expected_label and v.evidence_label and not v.has_dataset_limitation
    ]
    label_completed_correct = sum(1 for v in label_completed_pool if v.is_label_correct)
    verdict_completed_pool = [
        v
        for v in verdicts
        if v.completed_all_phases and v.expected_conclusion and v.verdict != "PARSE_ERROR" and not v.has_dataset_limitation
    ]
    verdict_completed_correct = sum(1 for v in verdict_completed_pool if v.is_correct)
    label_completed_non_l0_pool = [v for v in label_completed_pool if (v.complexity_tier or "").upper() != "L0"]
    label_completed_non_l0_correct = sum(1 for v in label_completed_non_l0_pool if v.is_label_correct)
    verdict_completed_non_l0_pool = [v for v in verdict_completed_pool if (v.complexity_tier or "").upper() != "L0"]
    verdict_completed_non_l0_correct = sum(1 for v in verdict_completed_non_l0_pool if v.is_correct)

    # Coding trial count aggregation
    trials_2a = [v.coding_trials_phase2a for v in verdicts if v.coding_trials_phase2a is not None]
    trials_2b = [v.coding_trials_phase2b for v in verdicts if v.coding_trials_phase2b is not None]
    trials_total = [v.coding_trials_total for v in verdicts if v.coding_trials_total is not None]
    coding_trials_stats = {
        "phase2a": {
            "n": len(trials_2a),
            "mean": (sum(trials_2a) / len(trials_2a)) if trials_2a else None,
            "max": max(trials_2a) if trials_2a else None,
        },
        "phase2b": {
            "n": len(trials_2b),
            "mean": (sum(trials_2b) / len(trials_2b)) if trials_2b else None,
            "max": max(trials_2b) if trials_2b else None,
        },
        "total": {
            "n": len(trials_total),
            "mean": (sum(trials_total) / len(trials_total)) if trials_total else None,
            "max": max(trials_total) if trials_total else None,
        },
    }

    # Phase-isolated reliability metrics (independent view per stage)
    phase1_pool = verdicts
    phase1_failures = [v for v in phase1_pool if v.failure_stage == "phase1"]
    phase1_schema_failures = [v for v in phase1_pool if v.failure_code == "P1_SCHEMA_INVALID"]
    phase1_semantic_obs_failures = [v for v in phase1_pool if v.failure_code == "P1_SEMANTIC_INVALID_OBSERVATIONS"]
    phase1_feasibility_declared = [
        v for v in phase1_pool if (v.phase1_feasibility_status or "").upper() in {"TESTABLE", "UNTESTABLE"}
    ]

    phase2a_pool = [
        v for v in verdicts
        if v.early_stop_type != "phase1_untestable" and v.failure_stage != "phase1"
    ]
    phase2a_failures = [v for v in phase2a_pool if v.failure_stage == "phase2a"]
    phase2a_missing_keys = [v for v in phase2a_pool if v.failure_code == "P2A_REQUEST_MISSING_KEYS"]
    phase2a_empty_identifiers = [v for v in phase2a_pool if v.failure_code == "P2A_EMPTY_IDENTIFIERS"]
    phase2a_observation_mismatch = [v for v in phase2a_pool if v.failure_code == "P2A_OBSERVATION_MISMATCH"]

    phase2b_pool = [
        v for v in verdicts
        if v.early_stop_type != "phase1_untestable" and v.failure_stage not in {"phase1", "phase2a"}
    ]
    phase2b_failures = [v for v in phase2b_pool if v.failure_stage == "phase2b"]
    phase2b_schema_failures = [v for v in phase2b_pool if v.failure_code in {"P2B_RESULTS_SCHEMA_INVALID", "P2B_RESULTS_MISSING"}]
    phase2b_synthetic_failures = [v for v in phase2b_pool if v.failure_code == "P2B_SYNTHETIC_DATA"]
    phase2b_sample_size_failures = [v for v in phase2b_pool if v.failure_code == "P2B_INVALID_SAMPLE_SIZE"]

    phase2b_results_pool = [
        v for v in verdicts
        if "phase2b_results_missing" not in set(v.warnings)
    ]
    phase2b_group_subset_violations = [v for v in phase2b_results_pool if v.phase2b_group_subset_violation is True]
    phase2b_missing_group_violations = [v for v in phase2b_results_pool if v.phase2b_missing_planned_groups]
    phase2b_n_total_sum_mismatch = [v for v in phase2b_results_pool if v.phase2b_n_total_sum_mismatch is True]
    phase2b_low_coverage = [v for v in phase2b_results_pool if "low_sample_utilization" in set(v.warnings)]
    phase2b_critical_coverage = [v for v in phase2b_results_pool if "critical_sample_utilization" in set(v.warnings)]
    phase2b_literal_p_assign = [v for v in phase2b_results_pool if v.phase2b_literal_pvalue_assignment]
    phase2b_boundary_p005 = [v for v in phase2b_results_pool if v.phase2b_boundary_pvalue_005]

    phase_metrics = {
        "phase1": {
            "pool": len(phase1_pool),
            "pass_count": len(phase1_pool) - len(phase1_failures),
            "pass_rate": ((len(phase1_pool) - len(phase1_failures)) / len(phase1_pool)) if phase1_pool else None,
            "schema_failure_count": len(phase1_schema_failures),
            "schema_failure_rate": (len(phase1_schema_failures) / len(phase1_pool)) if phase1_pool else None,
            "semantic_observation_failure_count": len(phase1_semantic_obs_failures),
            "semantic_observation_failure_rate": (len(phase1_semantic_obs_failures) / len(phase1_pool)) if phase1_pool else None,
            "feasibility_declared_count": len(phase1_feasibility_declared),
            "feasibility_declared_rate": (len(phase1_feasibility_declared) / len(phase1_pool)) if phase1_pool else None,
        },
        "phase2a": {
            "pool": len(phase2a_pool),
            "pass_count": len(phase2a_pool) - len(phase2a_failures),
            "pass_rate": ((len(phase2a_pool) - len(phase2a_failures)) / len(phase2a_pool)) if phase2a_pool else None,
            "request_schema_failure_count": len(phase2a_missing_keys),
            "request_schema_failure_rate": (len(phase2a_missing_keys) / len(phase2a_pool)) if phase2a_pool else None,
            "empty_identifier_failure_count": len(phase2a_empty_identifiers),
            "empty_identifier_failure_rate": (len(phase2a_empty_identifiers) / len(phase2a_pool)) if phase2a_pool else None,
            "observation_mismatch_count": len(phase2a_observation_mismatch),
            "observation_mismatch_rate": (len(phase2a_observation_mismatch) / len(phase2a_pool)) if phase2a_pool else None,
        },
        "phase2b": {
            "pool": len(phase2b_pool),
            "pass_count": len(phase2b_pool) - len(phase2b_failures),
            "pass_rate": ((len(phase2b_pool) - len(phase2b_failures)) / len(phase2b_pool)) if phase2b_pool else None,
            "results_schema_failure_count": len(phase2b_schema_failures),
            "results_schema_failure_rate": (len(phase2b_schema_failures) / len(phase2b_pool)) if phase2b_pool else None,
            "synthetic_failure_count": len(phase2b_synthetic_failures),
            "synthetic_failure_rate": (len(phase2b_synthetic_failures) / len(phase2b_pool)) if phase2b_pool else None,
            "invalid_sample_size_failure_count": len(phase2b_sample_size_failures),
            "invalid_sample_size_failure_rate": (len(phase2b_sample_size_failures) / len(phase2b_pool)) if phase2b_pool else None,
        },
    }

    # Evidence grounding: fraction of ALL runs with all 4 core statistical fields present
    _core_missing_fields = {"p_value_missing", "effect_size_missing", "test_used_missing", "sample_sizes_missing"}
    evidence_grounded = [v for v in verdicts if not _core_missing_fields.intersection(set(v.warnings))]
    evidence_grounding_rate = len(evidence_grounded) / len(verdicts) if verdicts else None

    data_curation_metrics = {
        "phase2b_results_pool": len(phase2b_results_pool),
        "group_subset_violation_count": len(phase2b_group_subset_violations),
        "group_subset_violation_rate": (len(phase2b_group_subset_violations) / len(phase2b_results_pool)) if phase2b_results_pool else None,
        "missing_planned_group_count": len(phase2b_missing_group_violations),
        "missing_planned_group_rate": (len(phase2b_missing_group_violations) / len(phase2b_results_pool)) if phase2b_results_pool else None,
        "n_total_sum_mismatch_count": len(phase2b_n_total_sum_mismatch),
        "n_total_sum_mismatch_rate": (len(phase2b_n_total_sum_mismatch) / len(phase2b_results_pool)) if phase2b_results_pool else None,
        "low_sample_coverage_count": len(phase2b_low_coverage),
        "low_sample_coverage_rate": (len(phase2b_low_coverage) / len(phase2b_results_pool)) if phase2b_results_pool else None,
        "critical_sample_coverage_count": len(phase2b_critical_coverage),
        "critical_sample_coverage_rate": (len(phase2b_critical_coverage) / len(phase2b_results_pool)) if phase2b_results_pool else None,
        "literal_pvalue_assignment_count": len(phase2b_literal_p_assign),
        "literal_pvalue_assignment_rate": (len(phase2b_literal_p_assign) / len(phase2b_results_pool)) if phase2b_results_pool else None,
        "boundary_pvalue_0_05_count": len(phase2b_boundary_p005),
        "boundary_pvalue_0_05_rate": (len(phase2b_boundary_p005) / len(phase2b_results_pool)) if phase2b_results_pool else None,
    }

    return {
        "total_runs": len(verdicts),
        "evidence_label_accuracy": (label_correct / label_total) if label_total > 0 else None,
        "evidence_label_correct": label_correct,
        "evidence_label_valid_runs": label_total,
        "evidence_label_accuracy_non_l0": (label_correct_non_l0 / label_total_non_l0) if label_total_non_l0 > 0 else None,
        "evidence_label_non_l0_correct": label_correct_non_l0,
        "evidence_label_non_l0_valid_runs": label_total_non_l0,
        "by_expected_label": by_expected_label if by_expected_label else None,
        "evidence_label_raw_accuracy": (raw_label_correct / raw_label_total) if raw_label_total > 0 else None,
        "evidence_label_raw_correct": raw_label_correct,
        "evidence_label_raw_valid_runs": raw_label_total,
        "evidence_label_raw_accuracy_non_l0": (raw_label_correct_non_l0 / raw_label_total_non_l0) if raw_label_total_non_l0 > 0 else None,
        "evidence_label_raw_non_l0_correct": raw_label_correct_non_l0,
        "evidence_label_raw_non_l0_valid_runs": raw_label_total_non_l0,
        "by_expected_label_raw": by_expected_label_raw if by_expected_label_raw else None,
        "verdict_accuracy": correct / total if total > 0 else 0,
        "verdict_correct": correct,
        "verdict_valid_runs": len(valid),
        "verdict_accuracy_non_l0": (correct_non_l0 / total_non_l0) if total_non_l0 > 0 else None,
        "verdict_non_l0_correct": correct_non_l0,
        "verdict_non_l0_valid_runs": total_non_l0,
        "by_expected_verdict": by_expected,
        "by_tier": by_tier if by_tier else None,
        "by_control_type": by_control_type if by_control_type else None,
        "verdict_mismatches": verdict_mismatches,
        "overclaim_rate": (overclaim_count / len(overclaim_pool)) if overclaim_pool else None,
        "overclaim_count": overclaim_count,
        "overclaim_pool": len(overclaim_pool),
        "false_refutation_rate": (false_refutation_count / len(predicted_refuted_pool)) if predicted_refuted_pool else None,
        "false_refutation_count": false_refutation_count,
        "false_refutation_pool": len(predicted_refuted_pool),
        "hallucinated_significance_rate": (hallucinated_sig_count / len(hallucinated_sig_pool)) if hallucinated_sig_pool else None,
        "hallucinated_significance_count": hallucinated_sig_count,
        "hallucinated_significance_pool": len(hallucinated_sig_pool),
        "synthetic_data_violation_rate": (synthetic_data_count / len(verdicts)) if verdicts else None,
        "synthetic_data_violation_count": synthetic_data_count,
        "synthetic_data_violation_pool": len(verdicts),
        "nonblocking_random_usage_rate": (random_usage_soft_count / len(verdicts)) if verdicts else None,
        "nonblocking_random_usage_count": random_usage_soft_count,
        "nonblocking_random_usage_pool": len(verdicts),
        "hallucination_any_rate": (broad_hallucination_count / len(broad_hallucination_pool)) if broad_hallucination_pool else None,
        "hallucination_any_count": broad_hallucination_count,
        "hallucination_any_pool": len(broad_hallucination_pool),
        "l0_feasibility_accuracy": (l0_feasibility_correct / len(l0_pool)) if l0_pool else None,
        "l0_feasibility_correct": l0_feasibility_correct,
        "l0_feasibility_pool": len(l0_pool),
        "l0_feasibility_accuracy_non_early_stops": (l0_non_early_correct / len(l0_non_early_pool)) if l0_non_early_pool else None,
        "l0_feasibility_correct_non_early_stops": l0_non_early_correct,
        "l0_feasibility_pool_non_early_stops": len(l0_non_early_pool),
        "early_untestable_stop_count": len(early_stop_pool),
        "early_untestable_stop_rate": (len(early_stop_pool) / len(verdicts)) if verdicts else None,
        "l0_early_untestable_stop_count": len(l0_early_stop_pool),
        "l0_early_untestable_stop_rate": (len(l0_early_stop_pool) / len(l0_pool)) if l0_pool else None,
        "l0_early_untestable_stop_correct": l0_early_stop_correct,
        "l0_early_untestable_stop_correct_rate": (l0_early_stop_correct / len(l0_early_stop_pool)) if l0_early_stop_pool else None,
        "l0_feasibility_miss_recovered_by_integrity_guard_count": l0_integrity_recovered_count,
        "l0_feasibility_miss_recovered_by_integrity_guard_rate": (l0_integrity_recovered_count / len(l0_pool)) if l0_pool else None,
        "non_l0_early_untestable_stop_count": len(non_l0_early_stop_pool),
        "parse_error_rows_total": parse_error_rows_total,
        "parse_runtime_failure_rows": parse_runtime_failure_rows,
        "parse_runtime_failure_rate": (parse_runtime_failure_rows / len(verdicts)) if verdicts else None,
        "parse_error_verdict_rows": parse_error_verdict_rows,
        "execution_completed_runs": completed_all_phases_rows,
        "execution_completed_rate": (completed_all_phases_rows / len(verdicts)) if verdicts else None,
        "evidence_grounding_rate": evidence_grounding_rate,
        "evidence_grounding_count": len(evidence_grounded),
        "evidence_grounding_pool": len(verdicts),
        "evidence_label_accuracy_e2e": (label_e2e_correct / len(label_e2e_pool)) if label_e2e_pool else None,
        "evidence_label_e2e_correct": label_e2e_correct,
        "evidence_label_e2e_pool": len(label_e2e_pool),
        "evidence_label_accuracy_e2e_non_l0": (label_e2e_non_l0_correct / len(label_e2e_non_l0_pool)) if label_e2e_non_l0_pool else None,
        "evidence_label_e2e_non_l0_correct": label_e2e_non_l0_correct,
        "evidence_label_e2e_non_l0_pool": len(label_e2e_non_l0_pool),
        "verdict_accuracy_e2e": (verdict_e2e_correct / len(verdict_e2e_pool)) if verdict_e2e_pool else None,
        "verdict_e2e_correct": verdict_e2e_correct,
        "verdict_e2e_pool": len(verdict_e2e_pool),
        "verdict_accuracy_e2e_non_l0": (verdict_e2e_non_l0_correct / len(verdict_e2e_non_l0_pool)) if verdict_e2e_non_l0_pool else None,
        "verdict_e2e_non_l0_correct": verdict_e2e_non_l0_correct,
        "verdict_e2e_non_l0_pool": len(verdict_e2e_non_l0_pool),
        "evidence_label_accuracy_completed": (label_completed_correct / len(label_completed_pool)) if label_completed_pool else None,
        "evidence_label_completed_correct": label_completed_correct,
        "evidence_label_completed_pool": len(label_completed_pool),
        "evidence_label_accuracy_completed_non_l0": (label_completed_non_l0_correct / len(label_completed_non_l0_pool)) if label_completed_non_l0_pool else None,
        "evidence_label_completed_non_l0_correct": label_completed_non_l0_correct,
        "evidence_label_completed_non_l0_pool": len(label_completed_non_l0_pool),
        "verdict_accuracy_completed": (verdict_completed_correct / len(verdict_completed_pool)) if verdict_completed_pool else None,
        "verdict_completed_correct": verdict_completed_correct,
        "verdict_completed_pool": len(verdict_completed_pool),
        "verdict_accuracy_completed_non_l0": (verdict_completed_non_l0_correct / len(verdict_completed_non_l0_pool)) if verdict_completed_non_l0_pool else None,
        "verdict_completed_non_l0_correct": verdict_completed_non_l0_correct,
        "verdict_completed_non_l0_pool": len(verdict_completed_non_l0_pool),
        "consistency": consistency if consistency else None,
        "consistency_summary": consistency_summary,
        "majority_vote": majority_vote,
        "majority_vote_e2e": majority_vote_e2e,
        "majority_vote_non_l0": majority_vote_non_l0,
        "majority_vote_e2e_non_l0": majority_vote_e2e_non_l0,
        "l0_feasibility_majority": l0_feasibility_majority,
        "l0_feasibility_majority_e2e": l0_feasibility_majority_e2e,
        "parse_methods": parse_methods,
        "failure_code_counts": failure_code_counts,
        "failure_stage_counts": failure_stage_counts,
        "phase2b_alignment": phase2b_counts,
        "phase_metrics": phase_metrics,
        "data_curation_metrics": data_curation_metrics,
        "warning_counts": warning_counts,
        "warning_counts_unique_runs": warning_counts_unique_runs,
        "coding_trials": coding_trials_stats,
    }


def print_summary(verdicts: List[WorkflowVerdict], metrics: Dict[str, Any]) -> None:
    """Print a human-readable summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if metrics.get("error"):
        print(f"\n{metrics['error']}")
        return

    print(f"\nTotal runs: {metrics['total_runs']}")

    # Primary metric: evidence label accuracy (deterministic from stats)
    label_acc = metrics.get("evidence_label_accuracy")
    label_valid = metrics.get("evidence_label_valid_runs", 0)
    if label_acc is not None:
        print(f"\nEvidence-label accuracy: {label_acc:.1%} ({metrics.get('evidence_label_correct', 0)}/{label_valid})")
    label_raw_acc = metrics.get("evidence_label_raw_accuracy")
    label_raw_valid = metrics.get("evidence_label_raw_valid_runs", 0)
    if label_raw_acc is not None:
        print(
            f"Evidence-label RAW accuracy: {label_raw_acc:.1%} "
            f"({metrics.get('evidence_label_raw_correct', 0)}/{label_raw_valid})"
        )
    if metrics.get("by_expected_label"):
        print("  Per-label breakdown:")
        for exp, data in metrics["by_expected_label"].items():
            print(f"    {exp}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
            print(f"      Predicted: {data['predicted']}")

    # Secondary metric: agent verdict accuracy (measures autonomy quality)
    verdict_valid = metrics.get("verdict_valid_runs", 0)
    if verdict_valid > 0:
        print(f"\nAgent verdict accuracy: {metrics['verdict_accuracy']:.1%} ({metrics.get('verdict_correct', 0)}/{verdict_valid})")
        if metrics.get("by_expected_verdict"):
            print("  Per-verdict breakdown:")
            for exp, data in metrics["by_expected_verdict"].items():
                print(f"    {exp}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
                print(f"      Predicted: {data['predicted']}")

    # Execution-aware reporting (separate runtime success from reasoning quality)
    completed_rate = metrics.get("execution_completed_rate")
    if completed_rate is not None:
        print(
            f"\nExecution completed-all-phases rate: {completed_rate:.1%} "
            f"({metrics.get('execution_completed_runs', 0)}/{metrics.get('total_runs', 0)})"
        )
    label_e2e = metrics.get("evidence_label_accuracy_e2e")
    if label_e2e is not None:
        print(
            f"Evidence-label accuracy (end-to-end): {label_e2e:.1%} "
            f"({metrics.get('evidence_label_e2e_correct', 0)}/{metrics.get('evidence_label_e2e_pool', 0)})"
        )
    verdict_e2e = metrics.get("verdict_accuracy_e2e")
    if verdict_e2e is not None:
        print(
            f"Agent verdict accuracy (end-to-end): {verdict_e2e:.1%} "
            f"({metrics.get('verdict_e2e_correct', 0)}/{metrics.get('verdict_e2e_pool', 0)})"
        )
    label_completed = metrics.get("evidence_label_accuracy_completed")
    if label_completed is not None:
        print(
            f"Evidence-label accuracy (completed runs): {label_completed:.1%} "
            f"({metrics.get('evidence_label_completed_correct', 0)}/{metrics.get('evidence_label_completed_pool', 0)})"
        )
    verdict_completed = metrics.get("verdict_accuracy_completed")
    if verdict_completed is not None:
        print(
            f"Agent verdict accuracy (completed runs): {verdict_completed:.1%} "
            f"({metrics.get('verdict_completed_correct', 0)}/{metrics.get('verdict_completed_pool', 0)})"
        )

    if metrics.get("verdict_mismatches", 0) > 0:
        print(f"\n  Verdict-evidence mismatches: {metrics['verdict_mismatches']} (agent verdict differs from evidence label)")

    if metrics.get("by_tier"):
        print("\nPer-tier metrics:")
        for tier, data in sorted(metrics["by_tier"].items()):
            label_acc = data.get("evidence_label_accuracy")
            verdict_acc = data.get("verdict_accuracy")
            label_str = f"{label_acc:.1%}" if isinstance(label_acc, float) else "N/A"
            verdict_str = f"{verdict_acc:.1%}" if isinstance(verdict_acc, float) else "N/A"
            print(
                f"  {tier}: label={label_str} ({data['label_correct']}/{data['label_valid']}), "
                f"verdict={verdict_str} ({data['verdict_correct']}/{data['verdict_valid']}), "
                f"runs={data['total_runs']}"
            )

    if metrics.get("by_control_type"):
        print("\nPer-control-type metrics:")
        for control, data in sorted(metrics["by_control_type"].items()):
            label_acc = data.get("evidence_label_accuracy")
            verdict_acc = data.get("verdict_accuracy")
            label_str = f"{label_acc:.1%}" if isinstance(label_acc, float) else "N/A"
            verdict_str = f"{verdict_acc:.1%}" if isinstance(verdict_acc, float) else "N/A"
            print(
                f"  {control}: label={label_str} ({data['label_correct']}/{data['label_valid']}), "
                f"verdict={verdict_str} ({data['verdict_correct']}/{data['verdict_valid']}), "
                f"runs={data['total_runs']}"
            )

    overclaim_rate = metrics.get("overclaim_rate")
    if overclaim_rate is not None:
        print(
            f"\nOverclaim rate: {overclaim_rate:.1%} "
            f"({metrics.get('overclaim_count', 0)}/{metrics.get('overclaim_pool', 0)})"
        )
    false_ref_rate = metrics.get("false_refutation_rate")
    if false_ref_rate is not None:
        print(
            f"False-refutation rate: {false_ref_rate:.1%} "
            f"({metrics.get('false_refutation_count', 0)}/{metrics.get('false_refutation_pool', 0)})"
        )
    hall_sig_rate = metrics.get("hallucinated_significance_rate")
    if hall_sig_rate is not None:
        print(
            f"Hallucinated-significance rate: {hall_sig_rate:.1%} "
            f"({metrics.get('hallucinated_significance_count', 0)}/{metrics.get('hallucinated_significance_pool', 0)})"
        )
    synth_rate = metrics.get("synthetic_data_violation_rate")
    if synth_rate is not None:
        print(
            f"Synthetic-data violation rate: {synth_rate:.1%} "
            f"({metrics.get('synthetic_data_violation_count', 0)}/{metrics.get('synthetic_data_violation_pool', 0)})"
        )
    soft_rng_rate = metrics.get("nonblocking_random_usage_rate")
    if soft_rng_rate is not None:
        print(
            f"Non-blocking random-usage rate: {soft_rng_rate:.1%} "
            f"({metrics.get('nonblocking_random_usage_count', 0)}/{metrics.get('nonblocking_random_usage_pool', 0)})"
        )
    hall_any_rate = metrics.get("hallucination_any_rate")
    if hall_any_rate is not None:
        print(
            f"Hallucination (broad) rate: {hall_any_rate:.1%} "
            f"({metrics.get('hallucination_any_count', 0)}/{metrics.get('hallucination_any_pool', 0)})"
        )
    eg_rate = metrics.get("evidence_grounding_rate")
    if eg_rate is not None:
        print(
            f"Evidence grounding rate: {eg_rate:.1%} "
            f"({metrics.get('evidence_grounding_count', 0)}/{metrics.get('evidence_grounding_pool', 0)})"
        )
    l0_rate = metrics.get("l0_feasibility_accuracy")
    if l0_rate is not None:
        print(
            f"L0 feasibility accuracy: {l0_rate:.1%} "
            f"({metrics.get('l0_feasibility_correct', 0)}/{metrics.get('l0_feasibility_pool', 0)})"
        )
        l0_early_rate = metrics.get("l0_early_untestable_stop_rate")
        if l0_early_rate is not None:
            print(
                f"  L0 early untestable stops: {l0_early_rate:.1%} "
                f"({metrics.get('l0_early_untestable_stop_count', 0)}/{metrics.get('l0_feasibility_pool', 0)})"
            )
        l0_non_early_rate = metrics.get("l0_feasibility_accuracy_non_early_stops")
        if l0_non_early_rate is not None:
            print(
                f"  L0 feasibility (excluding early stops): {l0_non_early_rate:.1%} "
                f"({metrics.get('l0_feasibility_correct_non_early_stops', 0)}/"
                f"{metrics.get('l0_feasibility_pool_non_early_stops', 0)})"
            )
        l0_recovered_rate = metrics.get("l0_feasibility_miss_recovered_by_integrity_guard_rate")
        if l0_recovered_rate is not None:
            print(
                f"  L0 feasibility misses recovered by integrity guard: {l0_recovered_rate:.1%} "
                f"({metrics.get('l0_feasibility_miss_recovered_by_integrity_guard_count', 0)}/"
                f"{metrics.get('l0_feasibility_pool', 0)})"
            )

    if metrics.get("consistency"):
        print("\nConsistency (repeated runs):")
        for hyp_id, data in metrics["consistency"].items():
            print(f"  {hyp_id}: {data['consistency_rate']:.1%} ({data['verdicts']})")
    if metrics.get("consistency_summary"):
        cs = metrics["consistency_summary"]
        mean_cons = cs.get("verdict_mean_consistency")
        if mean_cons is not None:
            print(
                f"\nConsistency summary: mean verdict consistency={mean_cons:.1%} "
                f"across {cs.get('verdict_hypotheses', 0)} repeated hypotheses"
            )
    if metrics.get("majority_vote"):
        mv = metrics["majority_vote"]
        ev = mv.get("evidence_label", {})
        vd = mv.get("verdict", {})
        if ev.get("total", 0):
            ev_acc = ev.get("accuracy")
            ev_str = f"{ev_acc:.1%}" if isinstance(ev_acc, float) else "N/A"
            print(
                "Hypothesis-level majority vote (evidence): "
                f"{ev_str} ({ev.get('correct', 0)}/{ev.get('total', 0)}), ties={ev.get('ties', 0)}"
            )
        if vd.get("total", 0):
            vd_acc = vd.get("accuracy")
            vd_str = f"{vd_acc:.1%}" if isinstance(vd_acc, float) else "N/A"
            print(
                "Hypothesis-level majority vote (verdict): "
                f"{vd_str} ({vd.get('correct', 0)}/{vd.get('total', 0)}), ties={vd.get('ties', 0)}"
            )
    if metrics.get("majority_vote_e2e"):
        mv = metrics["majority_vote_e2e"]
        ev = mv.get("evidence_label", {})
        vd = mv.get("verdict", {})
        if ev.get("total", 0):
            ev_acc = ev.get("accuracy")
            ev_str = f"{ev_acc:.1%}" if isinstance(ev_acc, float) else "N/A"
            print(
                "Hypothesis-level majority vote E2E (evidence): "
                f"{ev_str} ({ev.get('correct', 0)}/{ev.get('total', 0)}), "
                f"ties={ev.get('ties', 0)}, exec-fail majorities={ev.get('execution_failure_majority', 0)}"
            )
        if vd.get("total", 0):
            vd_acc = vd.get("accuracy")
            vd_str = f"{vd_acc:.1%}" if isinstance(vd_acc, float) else "N/A"
            print(
                "Hypothesis-level majority vote E2E (verdict): "
                f"{vd_str} ({vd.get('correct', 0)}/{vd.get('total', 0)}), "
                f"ties={vd.get('ties', 0)}, exec-fail majorities={vd.get('execution_failure_majority', 0)}"
            )

    print(f"\nParse methods: {metrics.get('parse_methods', {})}")
    if metrics.get("failure_code_counts"):
        print(f"Failure codes: {metrics.get('failure_code_counts')}")
    if metrics.get("failure_stage_counts"):
        print(f"Failure stages: {metrics.get('failure_stage_counts')}")
    if metrics.get("early_untestable_stop_count", 0) or metrics.get("parse_runtime_failure_rows", 0):
        prf = metrics.get("parse_runtime_failure_rate")
        prf_str = f"{prf:.1%}" if isinstance(prf, float) else "N/A"
        print(
            "Execution reliability: "
            f"early feasibility stops={metrics.get('early_untestable_stop_count', 0)}, "
            f"parse/runtime failures={metrics.get('parse_runtime_failure_rows', 0)} ({prf_str}), "
            f"PARSE_ERROR verdicts={metrics.get('parse_error_verdict_rows', 0)}"
        )

    if metrics.get("phase2b_alignment"):
        align = metrics["phase2b_alignment"]
        print(f"Phase 2B consistency: {align.get('consistent', 0)} consistent, {align.get('inconsistent', 0)} inconsistent, {align.get('unknown', 0)} unknown")

    if metrics.get("phase_metrics"):
        pm = metrics["phase_metrics"]
        print("\nPhase reliability:")
        for phase_name in ("phase1", "phase2a", "phase2b"):
            data = pm.get(phase_name, {})
            pass_rate = data.get("pass_rate")
            pass_str = f"{pass_rate:.1%}" if isinstance(pass_rate, float) else "N/A"
            print(
                f"  {phase_name}: pass={pass_str} "
                f"({data.get('pass_count', 0)}/{data.get('pool', 0)})"
            )

    if metrics.get("data_curation_metrics"):
        dc = metrics["data_curation_metrics"]
        pool = dc.get("phase2b_results_pool", 0)
        print("\nPhase 2B data curation:")
        print(
            "  "
            f"group-subset violations={dc.get('group_subset_violation_count', 0)}/{pool}, "
            f"missing planned groups={dc.get('missing_planned_group_count', 0)}/{pool}, "
            f"n_total/sample mismatch={dc.get('n_total_sum_mismatch_count', 0)}/{pool}"
        )
        print(
            "  "
            f"low coverage={dc.get('low_sample_coverage_count', 0)}/{pool}, "
            f"critical coverage={dc.get('critical_sample_coverage_count', 0)}/{pool}, "
            f"literal p-value assignment={dc.get('literal_pvalue_assignment_count', 0)}/{pool}, "
            f"p=0.05 boundary={dc.get('boundary_pvalue_0_05_count', 0)}/{pool}"
        )

    warnings = [v for v in verdicts if v.warnings]
    if warnings:
        print("\nWarnings:")
        for v in warnings:
            print(f"  {v.hypothesis_id}/{v.run_id}: {v.warnings}")

    if metrics.get("warning_counts"):
        print(f"\nWarning counts (raw): {metrics['warning_counts']}")
    if metrics.get("warning_counts_unique_runs"):
        print(f"Warning counts (unique runs): {metrics['warning_counts_unique_runs']}")

    ct = metrics.get("coding_trials", {})
    if ct.get("total", {}).get("n", 0) > 0:
        t = ct["total"]
        a = ct.get("phase2a", {})
        b = ct.get("phase2b", {})
        print(
            f"\nCoding trials ({t['n']} runs): "
            f"mean={t['mean']:.1f}, max={t['max']} total | "
            f"Phase 2A mean={a.get('mean', 0) or 0:.1f} | "
            f"Phase 2B mean={b.get('mean', 0) or 0:.1f}"
        )

    errors = [v for v in verdicts if v.verdict == "PARSE_ERROR"]
    if errors:
        print(f"\nParse errors ({len(errors)}):")
        for e in errors:
            print(f"  {e.hypothesis_id}/{e.run_id}: {e.parse_error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate workflow outputs")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing workflow outputs")
    parser.add_argument("--hypothesis-bank", type=Path, required=True,
                        help="Path to hypothesis_bank.json")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    sesoi_group = parser.add_mutually_exclusive_group()
    sesoi_group.add_argument(
        "--sesoi-profile",
        choices=["strict", "standard", "loose"],
        help="Override SESOI profile for all hypotheses during evaluation",
    )
    sesoi_group.add_argument(
        "--sesoi-value",
        type=float,
        help="Override SESOI with a fixed numeric value for all test families",
    )
    args = parser.parse_args()

    # Load hypothesis bank
    with open(args.hypothesis_bank) as f:
        bank = json.load(f)

    sesoi_override = args.sesoi_value if args.sesoi_value is not None else args.sesoi_profile

    # Run evaluation
    verdicts = evaluate_batch(
        args.results_dir,
        bank["hypotheses"],
        dataset_registry=bank.get("dataset_registry"),
        output_file=args.output,
        sesoi_override=sesoi_override,
    )

    # Compute and print metrics
    metrics = compute_metrics(verdicts)
    print_summary(verdicts, metrics)
