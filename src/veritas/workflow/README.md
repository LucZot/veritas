# Workflow Engine

## Overview

Dataset-driven multi-agent hypothesis testing framework. Users provide:
- A scientific hypothesis (natural language)
- A dataset with metadata (patient counts, cohorts, modality)

The workflow autonomously conducts end-to-end hypothesis validation through 4 specialized phases.

**Key design:**
- Agents discover dataset size via API calls (`list_dataset_patients()`)
- General plan schema supports any hypothesis type (not just case-control comparisons)
- Dataset metadata provides structure info (groups, observations, metadata fields)
- Works with ANY dataset — not domain-specific

## Quick Start

### 1. Basic Usage

```bash
# Run with default configuration
python src/veritas/workflow/runner.py

# Run with custom configuration
python src/veritas/workflow/runner.py -c src/veritas/workflow/configs/default.json

# Batch experiments (recommended)
python experiments/run_experiments.py --hypothesis-numbers 1 2 3
```

### 2. Configuration File

Create a JSON config specifying hypothesis, dataset, and agents.

## Workflow Phases

### Phase 1: Hypothesis Planning
- Input: Hypothesis + Dataset metadata
- Team: PI, Imaging Specialist, Statistician
- Output: Analysis plan JSON with groups, structures, observations, metrics, statistical_test, and optional analysis_type/predictors

**Plan JSON Schema (required + optional):**
```json
{
  "feasibility": {
    "status": "TESTABLE|UNTESTABLE",
    "invalid_subtype": "UNTESTABLE_MISSING_STRUCTURE|UNTESTABLE_MISSING_METADATA_FIELD|UNTESTABLE_MISSING_MODALITY|UNTESTABLE_MISSING_MEASUREMENT|UNTESTABLE_OTHER",
    "reason": "short reason when untestable",
    "missing_requirements": ["..."]
  },
  "groups": ["group1", "group2", ...],
  "structures": ["structure1", "structure2"],
  "observations": ["obs1", "obs2"],
  "metrics": ["metric1", ...],
  "statistical_test": "test name",
  "analysis_type": "group_difference|correlation|regression",
  "predictors": ["metadata_field", ...]
}
```

If `feasibility.status` is `UNTESTABLE`, the workflow stops after Phase 1 and writes a final verdict in `workflow_config.json` with `verdict=INCONCLUSIVE` and `evidence_label=INVALID`.

### Phase 2A: Imaging Analysis
- Input: Dataset + Analysis plan from Phase 1
- Agent: Medical Imaging Specialist (writes code)
- Output: Segmentation results

**Execution Flow:**
1. Agent writes `segmentation_request.json` to workspace
2. Script reads request and executes segmentation via MCP in main process
3. Results saved to shared cache database

**Segmentation Request Schema:**
```json
{
  "identifiers": ["dataset:patient001:obs1", "dataset:patient001:obs2", "..."],
  "structures": ["structure1", "structure2"],
  "results_database": "/path/to/sat_cache",
  "modality": "mri|ct|pet",
  "model_variant": "nano|pro",
  "chunk_size": 20
}
```

### Phase 2B: Statistical Analysis
- Input: Segmentation results from Phase 2A + plan from Phase 1
- Agent: Statistician (writes code)
- Output: Statistical results JSON, visualizations

**Required Output (`workspace/data/statistical_results.json`):**
```json
{
  "analysis_type": "group_difference|correlation|regression",
  "test_performed": "test name",
  "p_value": 0.001,
  "effect_size": 1.2,
  "effect_size_type": "cohens_d|rank_biserial|pearson_r|other",
  "n_total": 38,
  "sample_sizes": {"group1": 20, "group2": 18},
  "group_statistics": {"group1": {"mean": 0.0, "std": 0.0}, "...": {...}},
  "confidence_interval": {"lower": 0.5, "upper": 1.9, "level": 0.95},
  "adjusted_for": [],
  "stratified_by": []
}
```
- `n_total`: Total sample count — required for power analysis in the evaluator
- For correlation/regression analyses, `group_statistics` can be omitted or set to null
- `effect_size_type` should match the test (e.g., `rank_biserial` for Mann-Whitney, `pearson_r` for correlation)
- `adjusted_for` / `stratified_by`: confounders and stratification variables used in the analysis

### Phase 3: Interpretation & Validation
- Input: Results from all previous phases
- Team: PI, Imaging Specialist, Statistician
- Output: Verdict JSON with final interpretation

**Verdict Schema:**
```json
{
  "verdict": "YES|NO|INCONCLUSIVE",
  "evidence_label": "SUPPORTED|REFUTED|UNDERPOWERED|INVALID",
  "p_value": 0.001,
  "effect_size": 1.2,
  "test_used": "test name",
  "sample_sizes": {"group1": 20, "group2": 18},
  "confidence": "high|medium|low",
  "reasoning": "one sentence conclusion"
}
```

## Output Structure

```
outputs/workflow/
└── {experiment_name}/
    ├── workflow_config.json
    ├── phase1_hypothesis_formulation/
    ├── phase2a_imaging_analysis/
    ├── phase2b_statistical_analysis/
    │   └── workspace/
    └── phase3_interpretation/
```

## Configuration Parameters

See `configs/default.json` for full example with all fields.

**Key parameters:**

- `hypothesis`: Research question to test
- `dataset_path`: Path to dataset (required)
- `dataset`: Dataset metadata with available_groups, available_observations, patient_metadata_fields, domain_notes
- `sat_cache_database`: Optional shared cache for segmentation results
- Model assignments for each agent
- Context length, temperature, verbosity

## Non-ACDC Datasets (Manifest Format)

For datasets beyond ACDC, include a manifest at the dataset root (`dataset_manifest.json`, `dataset_index.json`, or `dataset.json`).
The manifest must list samples and their observation-to-path mapping so dataset tools and SAT can resolve identifiers.

Example:
```json
{
  "name": "my_dataset",
  "samples": [
    {
      "sample_id": "patient001",
      "group": "case",
      "observations": {
        "T0": "patient001/T0.nii.gz",
        "T1": "patient001/T1.nii.gz"
      }
    }
  ]
}
```

## Next Steps

1. Prepare dataset metadata for your hypothesis
2. Create config file with hypothesis, dataset_path, and metadata
3. Run `python src/veritas/workflow/runner.py -c your_config.json`
4. Monitor output in `outputs/workflow/{experiment_name}/`
5. Review Phase 3 summary for clinical interpretation
