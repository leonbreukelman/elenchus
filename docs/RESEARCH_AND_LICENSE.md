# Research Scope, Safety, and Licensing

## What this repo is

Elenchus is an experimental research prototype for testing whether model reasoning tracks changed constraints ("hard-to-vary" validation via council + probe).

## What this repo is not

- Not production software
- Not a safety-certified system
- Not a guaranteed source of mathematically correct outputs

## Required warning language

Use this wording consistently in public docs:

- "Research software. Use at your own risk."
- "May be incorrect, unstable, or misleading."
- "Do not use in production."

## Operator workflow (current)

```bash
# 1) Discover dataset options
uv run python scripts/benchmark_probe.py --list-datasets

# 2) Smoke test
uv run python scripts/benchmark_probe.py --dataset builtin --limit 5 --concurrency 1

# 3) Official benchmark lane
uv run python scripts/benchmark_probe.py --preset official-core --split train --limit 50 --output benchmark_official_core.json

# 4) Delta vs prior run
uv run python scripts/benchmark_probe.py --preset official-core --compare-to benchmark_official_core.json --output benchmark_official_core_compare.json
```

## License recommendation

### If you must block production use

Recommended: **PolyForm Strict 1.0.0**.

Reason: it is designed for non-production use, which matches the current research-only intent.

### If you want broad open-source adoption

Recommended: **Apache-2.0**.

Tradeoff: it does not stop production use; you can only warn, not prohibit.

## Practical next step

Choose one path and commit a top-level license file in this repository:

- `LICENSE` with PolyForm Strict 1.0.0 text, or
- `LICENSE` with Apache-2.0 text.

Until then, legal reuse terms remain ambiguous.