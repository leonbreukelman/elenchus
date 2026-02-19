# Elenchus

⚠️ Research project. Use at your own risk.

Elenchus verifies reasoning, not just answers.

Core test: change the constraints and measure whether the mechanism updates correctly.

## Research Status and Risk

- This repository is experimental research software and may be wrong, unstable, or misleading.
- Outputs are not safety-assured and are not suitable for production decision-making.
- Do not deploy this system in production environments.
- No warranty is provided for correctness, fitness, or reliability.

Pipeline:
1. Council solves in parallel (algebraic, numerical, symbolic) and forms consensus.
2. Deutsch Probe perturbs constraints, recomputes ground truth, and scores alignment.

## Setup

```bash
uv sync --dev
uv run pytest tests/ -v
```

## Configuration

Model routing is provider-agnostic through LiteLLM.

```bash
# Required
ELENCHUS_MODEL_FAST=openrouter/qwen/qwen3-32b
ELENCHUS_MODEL_CAPABLE=openrouter/deepseek/deepseek-r1-0528

# Optional token budgets
ELENCHUS_MAX_TOKENS_FAST=4096
ELENCHUS_MAX_TOKENS_CAPABLE=16384
```

For large reasoning models, set `ELENCHUS_MAX_TOKENS_CAPABLE=32768`.

## Benchmark

Built-in dataset: 85 problems from GSM8K, MATH, and curated sets.

```bash
# Discover available dataset and preset options
uv run python scripts/benchmark_probe.py --list-datasets

# Fast smoke test
uv run python scripts/benchmark_probe.py --dataset builtin --limit 5 --concurrency 1

# Single official dataset
uv run python scripts/benchmark_probe.py --concurrency 2 --output benchmark_results.json
uv run python scripts/benchmark_probe.py --dataset gsm8k --split train --limit 50

# Official core preset (GSM8K + MATH)
uv run python scripts/benchmark_probe.py --preset official-core --split train --limit 50

# Compare against a previous run
uv run python scripts/benchmark_probe.py --preset official-core --compare-to benchmark_results.json --output benchmark_compare.json
```

### Benchmark UX Notes

- `--dataset-path <file.json|file.jsonl>` loads your own dataset file.
- `--preset official-core` merges public benchmark sources (`gsm8k`, `math`).
- `--compare-to <baseline.json>` prints metric deltas vs. a previous result file.

## License Guidance

This repo currently has no committed project license file.

If you want to allow research/evaluation usage while prohibiting production use, a strong fit is **PolyForm Strict 1.0.0**.

If you want permissive open-source adoption instead, use **Apache-2.0** (but note: permissive licenses do **not** prevent production use).

See `docs/RESEARCH_AND_LICENSE.md` for practical guidance and tradeoffs.

## Development

- Test: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`
- Pre-commit: `ruff` + `gitleaks`

## What Changed This Week

### 2026-02-19

- Added first-class benchmark dataset UX:
	- `--dataset`, `--split`, `--dataset-path`
	- `--preset official-core` (GSM8K + MATH)
	- `--compare-to` for baseline delta reporting
- Added explicit research-only / non-production guidance in docs and notebook.
- Added `docs/RESEARCH_AND_LICENSE.md` with practical license tradeoffs.
- Fixed calibrated councilor async safety by using task-local DSPy context.
- Expanded tests for dataset loading, benchmark CLI helpers, and calibrated councilor paths.
