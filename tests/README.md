# Tests

Smoke tests for VERITAS.

## Import check

```bash
python -c "from veritas import run_meeting, Agent; from veritas.workflow.runner import main; print('OK')"
```

## Evaluation smoke test

Re-score the committed fixture without any LLM calls:

```bash
python experiments/evaluate_experiment_folder.py example_run/or_gpt52/
python experiments/evaluate_experiment_folder.py example_run/ollama_local/
```

## Full pipeline test (requires API key + data)

```bash
export OPENROUTER_API_KEY=sk-or-...
python experiments/run_experiments.py \
  --bank experiments/tiered_hypothesis_bank.json \
  --hypotheses cardiac_01_dcm_lvef_lower \
  --n-runs 1 \
  --config experiments/configs/default_openrouter_gpt52.json
```
