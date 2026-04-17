# Contributing to VERITAS

Thanks for your interest! VERITAS is actively developed — bug reports, reproduction attempts, and new dataset adapters are all welcome.

## Dev setup

```bash
git clone https://github.com/LucZot/veritas.git
cd veritas
conda create -n veritas python=3.12 -y
conda activate veritas
pip install -e ".[all]"
cp mcp_servers.example.json mcp_servers.json
```

Set `BIO_DATA_ROOT`, then either start Ollama or export `OPENROUTER_API_KEY` (see [README.md](README.md) for the full setup).

## Smoke test

```bash
# Offline fixture check (no LLM, no data)
python experiments/evaluate_experiment_folder.py example_run/ollama_local/

# Import check
python -c "from veritas.workflow.runner import main; from veritas import Agent, run_meeting"
```

## Where things live

| Area | Path |
|------|------|
| 4-phase pipeline | `src/veritas/workflow/` |
| Agent + meeting primitives | `src/veritas/agent.py`, `src/veritas/meetings/` |
| LangGraph execution | `src/veritas/graph/` |
| Prompts + agendas | `src/veritas/prompts.py`, `src/veritas/prompt_templates.py` |
| SAT / imaging API | `src/bio_api/`, `src/veritas/vision/` |
| MCP servers | `mcp_servers/code_execution/`, `mcp_servers/sat_segmentation/` |
| Benchmark + evaluator | `experiments/` |

## Pull request expectations

- Keep PRs scoped — a prompt rewrite and a new metric is two PRs.
- Include a one-line motivation in the description (the "why"; the diff covers the "what").
- If you change verdict-extraction or ECO logic: run against `example_run/ollama_local/` and paste the before/after metrics in the PR.
- Don't commit `outputs/`, `results/`, `tmp/`, or `mcp_servers.json` (all gitignored).

## Scope

We're happy to take:

- New dataset adapters (as long as there's a demonstrable hypothesis to run on them)
- Prompt / agenda improvements with a measurable delta on the 64-hypothesis benchmark
- Additional MCP-exposed analysis tools
- Bug fixes and reproducibility patches

## Reporting bugs

Use the bug-report template. Attaching artifcats that show failures makes triage much faster.
