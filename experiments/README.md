# VERITAS Experiments

Runner and evaluator for the 64-hypothesis benchmark (32 ACDC + 32 UCSF-PDGM). See [README.md](../README.md) for setup and running instructions.

## Flags

| Option | Description | Default |
|--------|-------------|---------|
| `--bank` | Hypothesis bank JSON | `experiments/tiered_hypothesis_bank.json` |
| `--hypotheses` | Hypothesis IDs to run | all in bank |
| `--hypothesis-range` | Numeric range like `1-8` or `1-4,7` | — |
| `--dataset` | Filter by dataset (`acdc`, `ucsf_pdgm`) | all |
| `--n-runs` | Repetitions per hypothesis | 3 |
| `--config` | Workflow config JSON | `src/veritas/workflow/configs/default.json` |
| `--output-dir` | Output directory | `results/experiments` |
| `--timeout` | Per-run timeout in seconds | 3600 |
| `--context-length` | Context window override | from config |
| `--no-evaluate` | Skip evaluation after running | — |

## Output structure

```
results/experiments/exp_YYYYMMDD_HHMMSS/
├── evaluation_results.json       # per-run verdicts + evidence labels
├── evaluation_metrics.json       # aggregate metrics
└── cardiac_01_dcm_lvef_lower/
    ├── run_000/
    │   ├── phase1_hypothesis_formulation/
    │   ├── phase2a_imaging_analysis/
    │   ├── phase2b_statistical_analysis/
    │   │   └── workspace/data/statistical_results.json
    │   └── phase3_interpretation/
    │       ├── discussion.md          # agent transcript + verdict
    │       └── validation_report.json
    └── run_001/
```

Re-evaluate any folder offline (no LLM calls):

```bash
python experiments/evaluate_experiment_folder.py results/.../exp_YYYYMMDD_HHMMSS
```

## Evaluation framework

Scores are computed deterministically from the Phase 2B statistical output — no post-hoc LLM grading.

Each run receives a **four-label evidence classification**:

| Label | Condition |
|-------|-----------|
| **SUPPORTED** | `p < 0.05` and direction matches hypothesis |
| **REFUTED** | `p ≥ 0.05` with power ≥ 0.80 at SESOI, or `p < 0.05` in the wrong direction |
| **UNDERPOWERED** | `p ≥ 0.05` with power < 0.80 at SESOI |
| **INVALID** | Execution failure, missing output, or logic contradiction |

**Power** is evaluator-computed (not agent-reported) using the realized N and a per-hypothesis smallest-effect-size-of-interest (SESOI). Each hypothesis declares a SESOI profile (`strict` / `standard` / `loose`) sized for the expected dataset scale. **Direction** is hard-coded in the bank's `meta_analysis` block — no NLP parsing of verdict text.

## Troubleshooting

- **GPU contention (Ollama + SAT on one GPU):** set `OLLAMA_KEEP_ALIVE=0` so Ollama unloads between calls, or run SAT on a second GPU.
- **Workflow timeouts:** raise `--timeout`; verify the SAT MCP server can launch (`mcp_servers.json` `sat` entry must point at a Python with SAT installed).
- **Observation names** must match exactly the strings in `available_observations` in the config (case-insensitive).
