# Elenchus

Elenchus verifies reasoning, not just answers.

Core test: change the constraints and measure whether the mechanism updates correctly.

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
uv run python scripts/benchmark_probe.py --concurrency 2 --output benchmark_results.json
```

## Development

- Test: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`
- Pre-commit: `ruff` + `gitleaks`
