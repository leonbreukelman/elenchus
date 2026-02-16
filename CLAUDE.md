# Elenchus

Neuro-symbolic math verification engine. Solves math problems with three parallel LLM strategies, reaches consensus, then verifies understanding via perturbation testing (the Deutsch Probe).

## Architecture

- `src/elenchus/engine.py` — Main LangGraph (router + council + consensus)
- `src/elenchus/probe/graph.py` — Independent probe subgraph
- `src/elenchus/state.py` — All shared data models
- `domains/` — YAML domain configs (domains are data, not code)

## Dev

- Package manager: uv
- Test: `uv run pytest`
- Lint: `uv run ruff check src/ tests/`
- Format: `uv run ruff format src/ tests/`

## Conventions

- LangGraph state: TypedDict
- Domain models: Pydantic BaseModel
- Config: Pydantic validated YAML
- Async throughout (LLM calls, sandbox execution)
- Structured logging via structlog
