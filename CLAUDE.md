# Elenchus

Neuro-symbolic math verification engine. Solves math problems with three parallel LLM strategies, reaches consensus, then verifies understanding via perturbation testing (the Deutsch Probe).

## Architecture

- `src/elenchus/engine.py` — Main LangGraph (router + council + consensus + conditional probe)
- `src/elenchus/probe/graph.py` — Deutsch Probe subgraph (extract → perturb → instruct → ground truth → score)
- `src/elenchus/probe/mechanism_judge.py` — LLM judge for reasoning quality (Haiku)
- `src/elenchus/calibration/` — DSPy prompt optimization pipeline
- `src/elenchus/state.py` — All shared data models
- `domains/` — YAML domain configs (domains are data, not code)

## Dev

- Package manager: uv
- Test: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`
- Calibrate: `uv run python scripts/calibrate.py --strategy numerical`
- Integration tests require `ANTHROPIC_API_KEY`

## Conventions

- LangGraph state: TypedDict
- Domain models: Pydantic BaseModel
- Config: Pydantic validated YAML
- Async throughout (LLM calls, sandbox execution)
- Structured logging via structlog
- Councilor seams via `_get_client()` for test mocking
- Calibration artifacts stored in `src/elenchus/calibration/artifacts/`

## CRITICALS

- Never hardcode API keys — always from environment
- Never import os/subprocess/shutil in sandboxed code
- Consensus tolerance is configurable via domain YAML, default 1e-3
- Ground truth validation: symbolic answer must match consensus before probe runs
