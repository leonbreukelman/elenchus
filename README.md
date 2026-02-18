# Elenchus

Neuro-symbolic math verification engine. Solves math problems with three parallel LLM strategies (algebraic, numerical, symbolic), reaches consensus, then verifies genuine understanding via perturbation testing — the Deutsch Probe.

## Setup

```bash
uv sync --dev
uv run pytest tests/ -v
```

Requires `ANTHROPIC_API_KEY` for integration tests.

## Configuration

Elenchus supports OpenRouter for cost-effective benchmarking. Create a `.env` file:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
ELENCHUS_MODEL_FAST=openrouter/qwen/qwen3-32b
ELENCHUS_MODEL_CAPABLE=openrouter/deepseek/deepseek-r1-0528
```

## Benchmarking

Run the Deutsch Probe benchmark against the built-in dataset (85 problems from GSM8K, MATH, and hand-curated sets).

```bash
# Run full benchmark (output to JSON)
uv run python scripts/benchmark_probe.py --concurrency 2 --output benchmark_results.json

# Run with watchdog (monitor for stalls)
uv run python scripts/watchdog.py &
```

## Architecture

Two LangGraph subgraphs:

1. **Council** — Three councilors solve the problem independently. Consensus engine compares answers within configurable tolerance (default 1e-3).

2. **Deutsch Probe** — Perturbs input constraints, has councilors instruct the model to find the new answer, computes ground truth via symbolic code, and scores alignment. An LLM judge evaluates mechanism quality (whether reasoning is mathematically coherent, not just numerically correct).

### Calibration

DSPy/MIPROv2 pipeline for offline prompt optimization:

```bash
uv run python scripts/calibrate.py --strategy numerical
uv run python scripts/calibrate.py --strategy algebraic
```

Optimized prompts are loaded at runtime automatically, falling back to hand-written defaults.

## Dev

- Package manager: `uv`
- Test: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`
- Pre-commit: ruff (lint + format) + gitleaks
