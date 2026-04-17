"""Template-driven prompt construction for phase agendas.

Agendas are built dynamically from dataset config and plan values,
not hardcoded with dataset-specific examples. This ensures the pipeline
works for any dataset without code changes.
"""

import json
from typing import Optional


def build_phase1_agenda(
    hypothesis: str,
    dataset_info_text: str,
    dataset_metadata: Optional[dict] = None,
) -> str:
    """Build Phase 1 (planning) discussion agenda.

    Args:
        hypothesis: The hypothesis text to test.
        dataset_info_text: Pre-formatted dataset info string.
        dataset_metadata: Raw dataset metadata dict for examples.
    """
    meta = dataset_metadata or {}
    groups = meta.get("available_groups", [])
    observations = meta.get("available_observations", [])

    return f"""**Research Question:** {hypothesis}

**Dataset Context:**
{dataset_info_text}

**Task:** produce one executable analysis plan.

**Core rules (all agents):**
- Keep the target quantity exact; no proxy substitution unless hypothesis explicitly allows it.
- If prerequisites are missing in principle, mark `UNTESTABLE` with subtype + missing requirements.
- Use only dataset observations/timepoints in `observations`; never place metadata there.

**Imaging specialist:**
- Choose exact structure names via `list_available_structures()`.
- Query cohort/sample sizes with `list_dataset_patients()` for planned groups.
- If hypothesis is metadata-only (no image-derived measurements), set `structures: []`.

**Statistician:**
- Choose one primary test that matches the hypothesis.
- Compute a priori power using queried sample sizes (`d=0.5` group tests, `r=0.3` correlations); report adequacy at 0.80.
- Survival: use `analysis_type: "survival"`; use log-rank for unadjusted two-group survival and Cox PH for adjusted/continuous-predictor survival.
- Mixed-cohort correlation/regression: consider confounding control via `adjust_for` or `stratify_by` (often `"group"`), especially when causal interpretation is intended.
- For correlation/regression, declare exact tested variables in `target_variables` (one outcome + predictors) using hypothesis terms.

**PI synthesis:**
- Use only metadata fields listed in dataset context.
- Metadata-value groups (e.g., GTR/STR) belong in `groups`; use `grouping_field` and `restrict_to` when needed.
- Derived groups require `group_spec: {{"type":"derived","rule":"..."}}`.

**Output:** a valid Phase 1 JSON plan block with feasibility + analysis contract fields (including `target_variables` for correlation/regression).
"""


def build_phase1_summary_instructions(
    dataset_metadata: Optional[dict] = None,
) -> str:
    """Build Phase 1 summary instructions for the PI.

    Args:
        dataset_metadata: Raw dataset metadata dict for dynamic examples.
    """
    meta = dataset_metadata or {}
    groups = meta.get("available_groups", [])
    observations = meta.get("available_observations", [])

    group_examples = json.dumps(groups[:2]) if groups else '["group1", "group2"]'
    obs_examples = json.dumps(observations[:2]) if observations else '["obs1", "obs2"]'

    return f"""
**CRITICAL — final output must end with exactly one JSON block (no text after it):**

```json
{{
  "feasibility": {{
    "status": "<TESTABLE|UNTESTABLE>",
    "invalid_subtype": "<UNTESTABLE_MISSING_STRUCTURE|UNTESTABLE_MISSING_METADATA_FIELD|UNTESTABLE_MISSING_MODALITY|UNTESTABLE_MISSING_MEASUREMENT|UNTESTABLE_OTHER>",
    "reason": "<short reason or null>",
    "missing_requirements": ["<item>", "..."]
  }},
  "cohort_mode": "<all|groups>",
  "groups": ["<group1>", "<group2>", "..."],
  "restrict_to": {{}},
  "structures": ["<structure1>", "..."],
  "observations": ["<obs1>", "..."],
  "metrics": ["<metric1>", "..."],
  "statistical_test": "<test name>",
  "analysis_type": "<group_difference|correlation|regression|survival>",
  "grouping_field": "<metadata field or null>",
  "group_spec": {{"type": "<dataset|metadata|derived|null>", "field": "<metadata field or null>", "rule": "<rule or null>"}},
  "predictors": ["<metadata_field>", "..."],
  "adjust_for": ["<metadata_field>", "..."],
  "stratify_by": ["<metadata_field>", "..."],
  "target_variables": {{
    "outcome": "<exact tested outcome variable>",
    "predictors": ["<exact tested predictor>", "..."]
  }}
}}
```

Rules:
- Feasibility first: if target cannot be tested in principle, set `UNTESTABLE` with subtype + missing requirements.
- For feasibility, check exact target quantity (no proxy substitution) and required modality/structures/observations/metadata.
- Groups:
  - Dataset cohort groups must use exact dataset labels (e.g., {group_examples}).
  - Metadata-value groups (e.g., GTR/STR, wildtype/mutant) go in `groups` and require `grouping_field`.
  - Derived groups require `group_spec.type="derived"` and a rule.
- Observations must be imaging observations/timepoints only, using exact labels (e.g., {obs_examples}); never metadata fields.
- Survival outcomes require `analysis_type="survival"`; use log-rank for unadjusted two-group survival and Cox PH for adjusted/continuous survival.
- Direct comparisons between named groups (higher/lower/more/less between groups) should use `analysis_type="group_difference"` with `predictors: []` unless explicit adjustment is part of the hypothesis.
- Correlation/regression:
  - `predictors` must be explicit hypothesis predictors and valid metadata fields.
  - Set `target_variables.outcome` + `target_variables.predictors` to the exact variables to be tested in Phase 2B.
  - Ensure variables match hypothesis terms (e.g., `ed_frame` vs `weight` is different from `height` vs `weight`).
  - In mixed cohorts (`cohort_mode="all"` or multiple groups), confounding control via `adjust_for` or `stratify_by` with `"group"` is recommended when scientifically warranted.
  - Use `adjust_for` only when you will run adjusted modeling in Phase 2B.
- Keep scope strict: no extra tests/covariates beyond hypothesis.
- If full cohort is intended, use `cohort_mode: "all"` and `groups: ["ALL"]`; otherwise list explicit groups.
- JSON must be valid and final.
"""


def build_phase2a_agenda(
    plan: dict,
    dataset_name: str,
    groups: list,
    structures: list,
    observations: list,
    results_db: str,
    modality: str,
    available_groups: list,
    metadata_fields: list,
    all_groups_selected: bool = False,
    metadata_groups: Optional[list] = None,
    cohort_mode: str = "",
    restrict_to: Optional[dict] = None,
) -> str:
    """Build Phase 2A (segmentation request) agenda.

    All examples use actual dataset values from config — no hardcoded defaults.
    """
    obs_example_1 = observations[0] if observations else "obs1"
    group_example = groups[0] if groups else "group1"
    metadata_groups = metadata_groups or []
    restrict_to = restrict_to or {}
    group_restrict_values: list[str] = []
    if "group" in restrict_to:
        raw_group = restrict_to.get("group")
        if isinstance(raw_group, (list, tuple, set)):
            group_restrict_values = [str(v).strip() for v in raw_group if str(v).strip()]
        elif raw_group is not None and str(raw_group).strip():
            group_restrict_values = [str(raw_group).strip()]
    grouping_field = plan.get("grouping_field")
    group_spec = plan.get("group_spec") if isinstance(plan.get("group_spec"), dict) else None
    if not grouping_field and group_spec and str(group_spec.get("type", "")).lower() == "metadata":
        grouping_field = group_spec.get("field")

    cohort_note = " (resolved from ALL groups)" if all_groups_selected else ""

    metadata_group_instructions = ""
    if metadata_groups:
        if grouping_field:
            if group_restrict_values:
                if len(group_restrict_values) == 1:
                    group_restrict_text = f'Combine with population restriction using group="{group_restrict_values[0]}".'
                else:
                    group_restrict_text = (
                        "Combine with population restriction by querying each cohort group separately "
                        f"(groups={group_restrict_values}) and merging identifiers."
                    )
                metadata_group_instructions = f"""
Plan groups {metadata_groups} are patient metadata values (not dataset group labels).
Use grouping_field="{grouping_field}" and query each group via metadata_filters={{{json.dumps(grouping_field)}: "<group_value>"}}.
First inspect distinct values for "{grouping_field}" from patient metadata, then use EXACT canonical values (no decorated labels like "IDH-wildtype" unless metadata truly contains that string).
{group_restrict_text}
"""
            else:
                metadata_group_instructions = f"""
Plan groups {metadata_groups} are patient metadata values (not dataset group labels).
Use grouping_field="{grouping_field}" and query each group via metadata_filters={{{json.dumps(grouping_field)}: "<group_value>"}}.
First inspect distinct values for "{grouping_field}" from patient metadata, then use EXACT canonical values (no decorated labels like "IDH-wildtype" unless metadata truly contains that string).
"""
        else:
            metadata_group_instructions = f"""
Plan groups {metadata_groups} are patient metadata values (not dataset group labels).
Find the metadata field containing these values and use metadata_filters to query them (see API reference below).
"""

    restrict_to_note = ""
    if restrict_to:
        api_parts = []
        if group_restrict_values:
            if len(group_restrict_values) == 1:
                api_parts.append(f'group="{group_restrict_values[0]}"')
            else:
                api_parts.append(
                    "for each group in "
                    f"{group_restrict_values}, call list_dataset_patients(\"{dataset_name}\", group=<value>) and merge"
                )
        meta_keys = {k: v for k, v in restrict_to.items() if k != "group"}
        if meta_keys:
            api_parts.append(f"metadata_filters={meta_keys}")
        api_hint = ", ".join(api_parts) if api_parts else str(restrict_to)
        restrict_to_note = (
            f"\n- **Population restriction**: {restrict_to} — include ONLY patients matching these criteria."
            f"\n  Apply cohort restriction in Phase 2A request building via: list_dataset_patients(\"{dataset_name}\", {api_hint})"
        )

    return f"""Build `segmentation_request.json` from the Phase 1 plan.

**Plan contract:**
- Dataset: {dataset_name}
- Groups: {groups}{cohort_note}
- Structures: {structures}
- Observations: {observations}
- Cohort mode: {cohort_mode or "groups"}{restrict_to_note}

**Rules:**
- Use exact structure names from plan (`{structures}`), unchanged.
- Build identifiers in format `{dataset_name}:<patient_id>:<observation>`.
- Include only plan observations: {observations}.
- If groups are `["ALL"]` / `cohort_mode="all"`, iterate real dataset groups; never pass `"ALL"` to the API.
- Group labels available: {list(available_groups)}
- Metadata fields available: {metadata_fields}
{metadata_group_instructions}
- For metadata-group hypotheses, map plan labels to canonical metadata values before querying.
- If any required group value cannot be mapped, stop and report it (do not output empty identifiers).

**API (pre-loaded):**
```python
result = list_dataset_patients("{dataset_name}", group="{group_example}")  # or metadata_filters={{...}}
meta = get_patient_metadata("{dataset_name}", "patient001")
for obs_name in {json.dumps(observations)}:
    identifier = meta["identifiers"].get(obs_name)
```

**Required output (`segmentation_request.json`):**
```json
{{
  "identifiers": ["{dataset_name}:patient001:{obs_example_1}", "..."],
  "structures": {json.dumps(structures)},
  "results_database": "{results_db}",
  "modality": "{modality}",
  "model_variant": "nano",
  "chunk_size": 64
}}
```

Write complete code now and save as `segmentation_request.json`.
"""


def build_phase2b_agenda(
    plan: dict,
    phase2a_results_db: str,
    groups: list,
    structures: list,
    observations: list,
    metrics: list,
    statistical_test: str,
    analysis_type: str,
    cohort_mode: str,
    predictors: list,
    adjust_for: list,
    stratify_by: list,
    metadata_fields: list,
    domain_notes: str,
    group_filter_instruction: str,
    target_variables: Optional[dict] = None,
    all_groups_selected: bool = False,
    restrict_to: Optional[dict] = None,
    adjust_for_level: str = "none",
    stratify_by_level: str = "none",
) -> str:
    """Build Phase 2B (statistical analysis) agenda.

    All examples use actual dataset values from config.
    """
    groups_str = ", ".join(groups) if groups else "ALL"
    group_1 = groups[0] if groups else "group1"
    group_2 = groups[1] if len(groups) > 1 else "group2"
    obs_example = observations[0] if observations else "obs1"
    structure_example = structures[0] if structures else "structure"

    is_metadata_only = len(structures) == 0
    requires_adjustment = bool(adjust_for)
    survival_mode = "non_survival"
    if analysis_type == "survival":
        survival_mode = "survival_adjusted" if requires_adjustment else "survival_unadjusted"
    elif analysis_type in {"correlation", "regression"} and requires_adjustment:
        survival_mode = "multivariate_adjusted"
    elif analysis_type in {"correlation", "regression"}:
        survival_mode = "multivariate_unadjusted"
    else:
        survival_mode = "group_difference"

    adjust_for_level_norm = str(adjust_for_level or "none").strip().lower()
    stratify_by_level_norm = str(stratify_by_level or "none").strip().lower()
    if adjust_for_level_norm not in {"required", "recommended", "none"}:
        adjust_for_level_norm = "none"
    if stratify_by_level_norm not in {"required", "recommended", "none"}:
        stratify_by_level_norm = "none"

    # Build explicit constraint strings for adjust_for and stratify_by
    if adjust_for:
        if adjust_for_level_norm == "required":
            adjust_str = f"{adjust_for} — REQUIRED: your code MUST adjust for these variables"
        elif adjust_for_level_norm == "recommended":
            adjust_str = f"{adjust_for} — RECOMMENDED: adjust for these variables when feasible"
        else:
            adjust_str = f"{adjust_for} — OPTIONAL: include only if statistically justified"
    else:
        adjust_str = "None — do NOT add covariates or adjustments"
    if stratify_by:
        if stratify_by_level_norm == "required":
            stratify_str = f"{stratify_by} — REQUIRED: your code MUST stratify by these variables"
        elif stratify_by_level_norm == "recommended":
            stratify_str = f"{stratify_by} — RECOMMENDED: stratify by these variables when feasible"
        else:
            stratify_str = f"{stratify_by} — OPTIONAL: include only if statistically justified"
    else:
        stratify_str = "None — do NOT stratify"

    # Restrict_to note (only when Phase 1 restricted the population)
    restrict_to = restrict_to or {}
    if restrict_to:
        restrict_to_note = (
            f"\n⚠️  Population restriction from Phase 1: {restrict_to} — "
            "load all patients with sat.list_patients(results_db_path), then filter by metadata/group exactly per this restriction.\n"
        )
    else:
        restrict_to_note = ""

    # Survival API note (only when relevant)
    if analysis_type == "survival":
        adjust_formula = f"group + {' + '.join(adjust_for)}" if adjust_for else "group"
        adjust_note = (
            f"\n  For adjusted survival ({adjust_for}): fit Cox PH with formula `{adjust_formula}`."
            if adjust_for else
            "\n  For unadjusted survival group comparisons: run log-rank and report Cox HR for the same grouping."
        )
        survival_api_note = (
            "\n- Survival API: use `survival_status` (1=death, 0=alive/censored) as the event indicator — never set all events=1 or all events=0."
            "\n  Store the hazard ratio from `cph.hazard_ratios_['group']` as `effect_size`, NOT the logrank test statistic."
            "\n  If CoxPHFitter produces HR=1.0 with CI=[1.0, 1.0], the fit failed — raise ValueError rather than storing degenerate output."
            + adjust_note + "\n"
        )
    else:
        survival_api_note = ""

    if survival_mode == "survival_adjusted":
        adjustment_line = (
            "- REQUIRED: Fit Cox PH with all plan covariates in `adjust_for`.\n"
            if adjust_for_level_norm == "required"
            else "- RECOMMENDED: Fit Cox PH with plan covariates in `adjust_for`.\n"
        )
        track_guidance = (
            "Track: survival_adjusted\n"
            + adjustment_line
            + "- REQUIRED: Report HR (effect_size_type=hazard_ratio), p-value, CI from the fitted Cox coefficient.\n"
            + (
                "- Forbidden: unadjusted-only log-rank result as final answer when adjustment is required.\n"
                if adjust_for_level_norm == "required"
                else "- Avoid reporting unadjusted-only log-rank as the sole final result.\n"
            )
            + "- Forbidden: substituting test statistics (e.g., chi-square) as hazard ratio.\n"
        )
    elif survival_mode == "survival_unadjusted":
        track_guidance = (
            "Track: survival_unadjusted\n"
            "- REQUIRED: Two-group log-rank for significance and Cox HR for effect size.\n"
            "- REQUIRED: Use full cohort after restrictions (no subsampling/caps).\n"
            "- Forbidden: t-test/Mann-Whitney for censored survival outcomes.\n"
        )
    elif survival_mode == "multivariate_adjusted":
        adjustment_line = (
            "- REQUIRED: Implement all `adjust_for` covariates in the fitted model.\n"
            if adjust_for_level_norm == "required"
            else "- RECOMMENDED: Implement planned `adjust_for` covariates in the fitted model.\n"
        )
        track_guidance = (
            "Track: multivariate_adjusted\n"
            + adjustment_line
            + "- REQUIRED: Report adjusted effect (not pooled unadjusted proxy).\n"
            + (
                "- Forbidden: declaring adjustment only in JSON without model implementation.\n"
                if adjust_for_level_norm == "required"
                else "- Avoid declaring adjustment only in JSON without model implementation.\n"
            )
        )
    else:
        track_guidance = (
            "Track: standard_group_or_unadjusted\n"
            "- REQUIRED: Apply only the planned test and planned cohort restrictions.\n"
            "- REQUIRED: Use all eligible patients (no manual sample caps).\n"
        )

    if is_metadata_only:
        task_steps = (
            "1. Load all patients via sat.list_patients(results_db_path)\n"
            f"2. {group_filter_instruction}\n"
            "3. Extract required metadata/outcomes per plan (no segmentation loading)\n"
            f"4. Perform {statistical_test} according to analysis_type={analysis_type}\n"
            "5. Create at least one plot in plots/*.png\n"
            "6. Save results to data/statistical_results.json"
        )
    else:
        task_steps = (
            "1. Load segmentation results for all patients using sat.list_patients(results_db_path)\n"
            f"2. {group_filter_instruction}\n"
            f"3. For each patient, load masks for structures {structures} at observations {observations}\n"
            f"4. Calculate metrics {metrics} from segmentation masks (use domain formulas)\n"
            "5. If analysis_type is correlation/regression, extract predictors from patient metadata\n"
            f"6. Perform {statistical_test} according to analysis_type={analysis_type}\n"
            "7. Create visualizations appropriate for the analysis\n"
            "8. Save results to data/statistical_results.json and plots/*.png"
        )

    output_contract_lines = []
    if analysis_type == "group_difference":
        output_contract_lines.append("For group-difference analyses, include `group_statistics` (per-group summary stats).")
    if adjust_for:
        if adjust_for_level_norm == "required":
            output_contract_lines.append(f"Include `adjusted_for` exactly as {adjust_for}.")
        elif adjust_for_level_norm == "recommended":
            output_contract_lines.append(f"Recommended: include `adjusted_for` as {adjust_for}.")
    if stratify_by:
        if stratify_by_level_norm == "required":
            output_contract_lines.append(f"Include `stratified_by` exactly as {stratify_by} and provide `stratified_results`.")
        elif stratify_by_level_norm == "recommended":
            output_contract_lines.append(f"Recommended: include `stratified_by` as {stratify_by} and provide `stratified_results`.")
    if analysis_type == "survival":
        output_contract_lines.append("For survival analyses, `effect_size_type` must be `hazard_ratio`.")
    output_contract_text = "\n".join(f"- {line}" for line in output_contract_lines)

    return f"""Execute Phase 2B statistical analysis using Phase 2A outputs.
{restrict_to_note}
**Plan contract (must match exactly):**
- results_db: {phase2a_results_db}
- groups: {groups_str}
- structures: {structures}
- observations: {observations}
- metrics: {metrics}
- statistical_test: {statistical_test}
- analysis_type: {analysis_type}
- cohort_mode: {cohort_mode or "groups"}
- predictors: {predictors if predictors else "None"}
- adjust_for: {adjust_str}
- stratify_by: {stratify_str}
- target_variables: {json.dumps(target_variables or {}, ensure_ascii=False)}
- available metadata fields: {metadata_fields if metadata_fields else "N/A"}

**Domain notes:** {domain_notes if domain_notes else "Use Phase 1 formulas and definitions."}

**Hard constraints:**
- Use SAT APIs as primary data source; no synthetic/mock placeholders.
- Use full eligible cohort after planned restrictions (no slicing/head/sample/max-patient caps).
- Prefer the planned primary analysis. If assumptions/data quality invalidate it, switch only to a statistically valid alternative and document the reason explicitly.
- Survival exception: companion outputs required by the plan are allowed (e.g., log-rank significance with Cox HR reporting).
- Implement planned adjustment/stratification in code and in `statistical_results.json`.
- If any planned group has n=0, raise an error and fix loading/filter logic.
- Before finishing, ensure `data/statistical_results.json` exists with required keys.
- For correlation/regression, include `variables_tested` with exact `outcome` and `predictors` used in code.

**Analysis track (strict):**
{track_guidance}

**Task:**
Write Python code to:
{task_steps}

The variable `results_db_path` is pre-loaded in your environment — use it directly, do NOT redefine it with a hardcoded path.

**SAT API (pre-loaded):**
- `sat.list_patients(results_db_path)` -> all patient IDs
- `sat.get_patient_metadata(patient_id)` -> metadata (group + clinical fields)
- `sat.get_observation_identifiers(patient_id)` -> per-patient observation identifiers
- `sat.load_structure_mask(results_db_path, patient_id, structure, source_image_contains=...)`
- `sat.calculate_volume(mask, spacing)`

Observation usage example:
```python
obs_map = sat.get_observation_identifiers(patient_id)
for obs_name in {json.dumps(observations)}:
    masks = sat.load_structure_mask(results_db_path, patient_id, '{structure_example}', source_image_contains=obs_map[obs_name])
```
{survival_api_note}
**Required output (`data/statistical_results.json`):**
```json
{{
  "analysis_type": "{analysis_type}",
  "test_performed": "{statistical_test}",
  "p_value": <float>,
  "effect_size": <float>,
  "effect_size_type": "<hazard_ratio|cohens_d|rank_biserial|pearson_r|spearman_rho|partial_pearson_r|regression_beta|other>",
  "n_total": <int>,
  "sample_sizes": {{"{group_1}": <int>, "{group_2}": <int>, "...": <int>}},
  "variables_tested": {{"outcome": "<variable>", "predictors": ["<variable>", "..."]}}
}}
```

**Additional output contract:**
{output_contract_text if output_contract_text else "- No additional fields are required beyond the core schema for this plan."}

Write complete analysis code now.
"""
