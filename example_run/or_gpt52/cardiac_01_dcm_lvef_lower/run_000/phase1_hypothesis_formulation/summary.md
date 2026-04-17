### Agenda

Design an executable Phase 1 analysis plan to test the hypothesis that **DCM patients have significantly lower left ventricular ejection fraction (LVEF)** than **normal controls (NOR)** in the **ACDC cardiac cine short-axis MRI** dataset, using **ED and ES** observations and the exact definition of LVEF:  
\[
\mathrm{LVEF} = \frac{\mathrm{LVEDV}-\mathrm{LVESV}}{\mathrm{LVEDV}} \times 100
\]

### Team Member Input

**Medical Imaging Specialist**
- Verified tooling readiness: SAT models/checkpoints are available and usable.
- Confirmed available cardiac structures via `list_available_structures(...)`; key exact structure name needed for LV volume is **"left heart ventricle"** (with **"myocardium"** also available but not required for LVEF).
- Queried cohort sizes using `list_dataset_patients()`:
  - **DCM n = 30**
  - **NOR n = 30**
- Recommended segmentation/measurement approach:
  - Use ED and ES frames only (align with allowed observations).
  - Compute LVEDV/LVESV by summing **slice mask area × through-plane spacing** across the short-axis stack.
  - Implement QC for basal/apical slices; exclude missing/failed cases; apply minimal area threshold to remove noise.

**ML Statistician**
- Proposed single primary test consistent with hypothesis: **two-sample independent comparison** of LVEF (DCM vs NOR), recommending **Welch’s two-sample t-test**, **two-sided**, α=0.05 as primary.
- Recommended diagnostics (secondary checks, not changing the primary test): distribution checks (Shapiro/QQ); optional Mann–Whitney as a robustness check if severe non-normality.
- Power analysis requirement addressed using queried sample sizes and rule-specified effect size **d=0.5**:
  - Code execution returned **power = 0.478** (underpowered vs 0.80 target).
  - Concluded current sample is underpowered for d=0.5; suggested ~63 per group for 0.80 power (approximate planning guidance).
- Suggested keeping primary analysis unadjusted; optional adjusted model as secondary only if needed (height/weight), but not required by the hypothesis.

### Recommendation

Proceed with a **testable, strict-scope** Phase 1 plan:

- **Primary endpoint:** subject-level **LVEF** computed exactly from **LVEDV and LVESV** derived at **ED and ES** from the **"left heart ventricle"** structure.
- **Primary analysis:** **Welch’s two-sample t-test (two-sided, α=0.05)** comparing LVEF between **DCM vs NOR**.
- **Scope discipline:** Do **not** add covariate adjustment or extra endpoints in the primary plan because the hypothesis is a simple group difference. (QC and exclusions are part of measurement validity, not hypothesis expansion.)
- **Interpretation stance:** The analysis is feasible and executable, but **underpowered at d=0.5** with n=30/group; therefore, treat inference as **estimative/exploratory** in terms of sensitivity, emphasizing **effect size and confidence intervals** alongside the p-value.

Justification:
- This matches the hypothesis exactly (DCM vs NOR difference in LVEF), uses only allowed observations (ED/ES), and relies on an appropriate, pre-specified single primary test. Power limitations are explicitly documented without altering the hypothesis or substituting proxy endpoints.

### Next Steps

1. Implement the imaging derivation pipeline:
   - Segment **"left heart ventricle"** at **ED** and **ES** (use provided masks if present; otherwise generate with SAT).
   - Compute LVEDV/LVESV by slice-wise integration using image geometry; compute LVEF exactly.
2. Apply QC/exclusion rules consistently and log excluded cases (missing ED/ES, failed segmentation, invalid volumes).
3. Run the primary statistical test (Welch t-test) and report mean difference, 95% CI, and Cohen’s d.
4. Document power limitation (0.478 at d=0.5) in the analysis report and decide separately (outside this hypothesis plan) whether to pursue additional data for confirmatory power.

```json
{
  "feasibility": {
    "status": "TESTABLE",
    "invalid_subtype": "UNTESTABLE_OTHER",
    "reason": null,
    "missing_requirements": []
  },
  "cohort_mode": "groups",
  "groups": ["DCM", "NOR"],
  "restrict_to": {},
  "structures": ["left heart ventricle"],
  "observations": ["ED", "ES"],
  "metrics": ["LVEF"],
  "statistical_test": "Welch two-sample t-test (independent, two-sided)",
  "analysis_type": "group_difference",
  "grouping_field": "group",
  "group_spec": {
    "type": "dataset",
    "field": "group",
    "rule": null
  },
  "predictors": [],
  "adjust_for": [],
  "stratify_by": [],
  "target_variables": {
    "outcome": "LVEF",
    "predictors": ["group (DCM vs NOR)"]
  }
}
```