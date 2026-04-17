```json
{
  "feasibility": {
    "status": "TESTABLE",
    "invalid_subtype": null,
    "reason": null,
    "missing_requirements": []
  },
  "cohort_mode": "groups",
  "groups": ["DCM", "NOR"],
  "restrict_to": {},
  "structures": ["left heart ventricle"],
  "observations": ["ED", "ES"],
  "metrics": ["LVEF"],
  "statistical_test": "two-sample t-test",
  "analysis_type": "group_difference",
  "grouping_field": "group",
  "group_spec": {
    "type": "dataset",
    "field": null,
    "rule": null
  },
  "predictors": [],
  "adjust_for": [],
  "stratify_by": [],
  "target_variables": {
    "outcome": "LVEF",
    "predictors": []
  }
}
```