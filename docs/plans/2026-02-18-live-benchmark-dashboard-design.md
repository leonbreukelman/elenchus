# Live Benchmark Dashboard

## Summary

A Rich-based live terminal dashboard for Elenchus benchmarks. Shows real-time results as problems complete, with running totals and active problem tracking.

## Architecture

New file: `scripts/benchmark_live.py`. Reuses the same pipeline logic as `benchmark_probe.py` (council, consensus, probe) but replaces batch `asyncio.gather()` with `asyncio.as_completed()` for streaming results. `rich` added to dev dependencies in `pyproject.toml`.

The existing `benchmark_probe.py` stays untouched. Both scripts work independently.

## Display Layout

Three stacked sections inside a Rich `Live` context:

### Header Panel

Shows run config and running totals that update on each completion:

- Models (fast + capable)
- Progress: `12/20`
- Running accuracy: `Correct: 5 (42%)`
- Error count
- Wall-clock elapsed time

### Results Table

Same columns as current output: `#`, `Ans` (checkmark/cross/ERR), `Verdict`, `Score`, `Mech`, `Pert`, `Time`, `Question`. Rows appear as problems finish.

Color coding:
- Green: correct answer + hard_to_vary
- Red: error
- Yellow: wrong answer
- Dim: easy_to_vary

### Active Panel

Shows currently running problems: problem number, question snippet (50 chars), live elapsed timer. Shrinks as problems finish, disappears when run completes. With concurrency=2, typically shows 2 lines.

## Data Flow

1. Script creates all problem tasks upfront
2. Wraps with `asyncio.as_completed()`
3. On each completion: add row to table, remove from active panel, recalculate header totals
4. Rich `Live` redraws after each update
5. Background task ticks active timers every second for live elapsed display

## Post-Completion

Live display stops (final state stays on screen). Static summary prints below: cross-tabulation table, per-category breakdown, JSON output path. Same output as current `benchmark_probe.py`.

## Error Handling

Per-problem exceptions caught and shown as "ERR" rows. Dashboard never crashes from individual problem failures.

## Dependencies

- `rich` added to `[project.optional-dependencies]` dev group (benchmark tooling, not runtime)

## CLI Interface

```
uv run python scripts/benchmark_live.py --limit 20 --concurrency 2 --output results.json
```

Same flags as `benchmark_probe.py`: `--limit`, `--category`, `--tolerance`, `--output`, `--concurrency`.
