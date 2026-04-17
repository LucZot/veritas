# Example VERITAS Runs

Two **committed, frozen examples** of a complete VERITAS run on the same hypothesis — inspect them without installing or running anything.

**Hypothesis:** *DCM patients show significantly lower LVEF than normal controls.* (ACDC cardiac MRI, 30 DCM vs 30 NOR subjects.)

Both runs reach the same verdict — `YES / SUPPORTED`, but use very different model stacks:

| Fixture | Provider | PI / imaging | Statistician (discussion / coding) | Critic |
|---|---|---|---|---|
| [`local/`](ollama_local/) | Local (Ollama) | `gpt-oss:20b` | `qwen3:8b` / `qwen3-coder:30b` | `qwen3:8b` |
| [`gpt_5.2/`](or_gpt52/) | OpenRouter | `gpt-5.2` | `gpt-5-mini` / `gpt-5.2` | `gpt-5-mini` |

The local VERITAS runs on a single workstation!

## Layout

Each fixture has the same output structure:

```
<fixture>/
├── evaluation_results.json           # per-run verdict + evidence label
├── evaluation_metrics.json           # aggregate metrics
└── cardiac_01_dcm_lvef_lower/
    └── run_000/
        ├── experiment_config.json    # hypothesis + dataset + phase toggles
        ├── workflow_config.json      # LLM provider, models, temperatures, phases
        ├── phase1_hypothesis_formulation/   # agent-authored analysis plan
        │   ├── discussion.md
        │   ├── discussion.json
        │   ├── summary.md
        │   └── plan_contract_audit.json
        ├── phase2a_imaging_analysis/        # code that called the SAT MCP server
        │   ├── discussion.md
        │   ├── workspace/code/build_request.py
        │   └── segmentation_execution.json
        ├── phase2b_statistical_analysis/    # code that ran the statistical test
        │   ├── discussion.md
        │   ├── workspace/code/analysis.py
        │   ├── workspace/plots/group_comparison.png
        │   └── workspace/data/statistical_results.json
        └── phase3_interpretation/           # team interpretation + verdict JSON
            ├── discussion.md
            ├── discussion.json
            └── validation_report.json
```

## What to look at first

- **`phase1_hypothesis_formulation/discussion.md`** — how the PI + critic + team lead shaped the analysis plan.
- **`phase2b_statistical_analysis/workspace/data/statistical_results.json`** — machine-readable test output (p-value, effect size, group stats, CI).
- **`phase2b_statistical_analysis/workspace/plots/*`** — agent-generated plots.
- **`phase3_interpretation/discussion.md`** — the full multi-agent interpretation debate, ending with the verdict JSON.

Reading the two `phase3_interpretation/discussion.md` files side by side is a good way to see how the cadence and verbosity of the debate changes between frontier and open-weight agents, even when the verdict is the same.

## Reproducing the evaluation

The evaluator runs offline (no LLM calls) — it parses the verdict JSON and scores it against ground truth:

```bash
python experiments/evaluate_experiment_folder.py example_run/or_gpt52/
python experiments/evaluate_experiment_folder.py example_run/ollama_local/
```

Each regenerates that fixture's `evaluation_results.json` and `evaluation_metrics.json` and prints the per-run verdict.
