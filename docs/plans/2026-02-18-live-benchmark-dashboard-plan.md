# Live Benchmark Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Rich-based live terminal dashboard (`scripts/benchmark_live.py`) that shows real-time benchmark results as problems complete.

**Architecture:** New script imports shared logic from `benchmark_probe.py` (dataclasses, `run_one_problem`, `build_summary`, `print_results`). Replaces `asyncio.gather()` with `asyncio.as_completed()` for streaming results. Rich `Live` context redraws three stacked panels on each completion. A background asyncio task ticks active-problem timers every second.

**Tech Stack:** `rich` (Live, Table, Panel, Layout, Text), asyncio, existing Elenchus pipeline.

---

### Task 1: Add `rich` to dev dependencies

**Files:**
- Modify: `pyproject.toml:22-28`

**Step 1: Add rich to the dev optional-dependencies group**

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6",
    "ruff>=0.15",
    "pre-commit>=4",
    "rich>=13",
]
```

**Step 2: Install the updated dependencies**

Run: `cd projects/elenchus && uv sync --extra dev`
Expected: rich installed, lock file updated

**Step 3: Verify rich is importable**

Run: `cd projects/elenchus && uv run python -c "import rich; print(rich.__version__)"`
Expected: prints version number (13.x)

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add rich to dev dependencies for live dashboard"
```

---

### Task 2: Create display helper functions with tests

These are the pure-logic rendering functions that build Rich renderables from data. Testing them in isolation keeps the main script integration-only.

**Files:**
- Create: `scripts/benchmark_display.py`
- Create: `tests/test_benchmark_display.py`

**Step 1: Write the failing tests**

```python
"""Tests for benchmark display helpers."""

import sys
from pathlib import Path

# Scripts aren't a package — add to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_display import ActiveProblem, make_header, make_results_table, make_active_panel, row_style
from benchmark_probe import ProbeProblemResult


def _ok_result(**overrides) -> ProbeProblemResult:
    """Factory for a completed problem result."""
    defaults = dict(
        question="A train travels 60 mph for 3 hours...",
        category="rate",
        source="gsm8k",
        expected=180.0,
        consensus_answer=180.0,
        agreement="unanimous",
        answer_correct=True,
        probe_verdict="hard_to_vary",
        probe_score=0.85,
        sensitivity_map={"speed": 0.9},
        mechanism_scores=[0.8, 0.7],
        num_perturbations=3,
        num_ground_truths_ok=2,
        elapsed_s=12.5,
        error=None,
    )
    defaults.update(overrides)
    return ProbeProblemResult(**defaults)


class TestRowStyle:
    def test_correct_hard_to_vary_is_green(self):
        assert row_style(_ok_result()) == "green"

    def test_error_is_red(self):
        r = _ok_result(error="boom", probe_verdict="error", answer_correct=False)
        assert row_style(r) == "red"

    def test_wrong_answer_is_yellow(self):
        r = _ok_result(answer_correct=False, probe_verdict="hard_to_vary")
        assert row_style(r) == "yellow"

    def test_easy_to_vary_is_dim(self):
        r = _ok_result(probe_verdict="easy_to_vary")
        assert row_style(r) == "dim"

    def test_partially_coupled_is_default(self):
        r = _ok_result(probe_verdict="partially_coupled")
        assert row_style(r) == ""


class TestMakeHeader:
    def test_shows_progress(self):
        panel = make_header(
            model_fast="gemini-flash", model_capable="gemini-pro",
            completed=5, total=20, correct=3, errors=1, elapsed=45.2,
        )
        text = panel.renderable.plain
        assert "5/20" in text
        assert "3 (60.0%)" in text
        assert "1" in text

    def test_zero_completed(self):
        panel = make_header(
            model_fast="fast", model_capable="capable",
            completed=0, total=10, correct=0, errors=0, elapsed=0.0,
        )
        text = panel.renderable.plain
        assert "0/10" in text


class TestMakeResultsTable:
    def test_empty_table_has_headers(self):
        table = make_results_table([])
        assert table.row_count == 0

    def test_one_result_adds_row(self):
        table = make_results_table([_ok_result()])
        assert table.row_count == 1

    def test_error_result_shows_err(self):
        table = make_results_table([_ok_result(error="fail", probe_verdict="error")])
        assert table.row_count == 1


class TestMakeActivePanel:
    def test_no_active_returns_empty_panel(self):
        panel = make_active_panel([])
        # Should still be a valid Panel
        assert panel is not None

    def test_shows_running_problems(self):
        active = [
            ActiveProblem(index=1, question="What is 2+2?", start_time=0.0),
            ActiveProblem(index=3, question="A train goes...", start_time=0.0),
        ]
        panel = make_active_panel(active, now=10.0)
        text = panel.renderable.plain
        assert "#1" in text or "1" in text
        assert "#3" in text or "3" in text
```

**Step 2: Run tests to verify they fail**

Run: `cd projects/elenchus && uv run pytest tests/test_benchmark_display.py -v`
Expected: FAIL — `benchmark_display` module not found

**Step 3: Implement the display helpers**

```python
"""Rich display helpers for the live benchmark dashboard.

Pure rendering functions — no I/O, no async. Each returns a Rich
renderable (Panel, Table) that the Live context composites.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import the result dataclass from the benchmark script.
# Both files live in scripts/, so this import works directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from benchmark_probe import ProbeProblemResult  # noqa: E402


@dataclass
class ActiveProblem:
    """A problem currently being processed."""

    index: int
    question: str
    start_time: float


def row_style(result: ProbeProblemResult) -> str:
    """Return a Rich style string for a result row."""
    if result.error:
        return "red"
    if not result.answer_correct:
        return "yellow"
    if result.probe_verdict == "easy_to_vary":
        return "dim"
    if result.probe_verdict == "hard_to_vary":
        return "green"
    return ""


def make_header(
    *,
    model_fast: str,
    model_capable: str,
    completed: int,
    total: int,
    correct: int,
    errors: int,
    elapsed: float,
) -> Panel:
    """Build the header panel with run config and running totals."""
    pct = f"{correct / completed * 100:.1f}%" if completed else "—"
    lines = [
        f"Models: {model_fast} / {model_capable}",
        f"Progress: {completed}/{total}    "
        f"Correct: {correct} ({pct})    "
        f"Errors: {errors}    "
        f"Elapsed: {elapsed:.0f}s",
    ]
    return Panel(Text("\n".join(lines)), title="Deutsch Probe Benchmark", border_style="blue")


def make_results_table(results: list[ProbeProblemResult]) -> Table:
    """Build the results table from completed results so far."""
    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("#", width=4, justify="right")
    table.add_column("Ans", width=3, justify="center")
    table.add_column("Verdict", width=16)
    table.add_column("Score", width=6, justify="right")
    table.add_column("Mech", width=6, justify="right")
    table.add_column("Pert", width=5, justify="right")
    table.add_column("Time", width=7, justify="right")
    table.add_column("Question", ratio=1, no_wrap=True, overflow="ellipsis")

    for i, r in enumerate(results, 1):
        style = row_style(r)

        ans = "ERR" if r.error else (" ✓" if r.answer_correct else " ✗")
        verdict = r.probe_verdict.upper() if r.probe_verdict == "hard_to_vary" else r.probe_verdict
        score = f"{r.probe_score:.2f}" if r.probe_score else "—"
        mech = "—"
        if r.mechanism_scores:
            mech = f"{sum(r.mechanism_scores) / len(r.mechanism_scores):.2f}"
        perts = f"{r.num_ground_truths_ok}/{r.num_perturbations}"

        table.add_row(
            str(i), ans, verdict, score, mech, perts,
            f"{r.elapsed_s:.1f}s", r.question[:50],
            style=style,
        )

    return table


def make_active_panel(active: list[ActiveProblem], *, now: float = 0.0) -> Panel:
    """Build the active-problems panel showing running problems with timers."""
    if not active:
        return Panel(Text("All problems completed.", style="dim"), title="Active", border_style="dim")

    lines: list[str] = []
    for ap in active:
        elapsed = now - ap.start_time if now else 0.0
        snippet = ap.question[:50]
        lines.append(f"  #{ap.index:<4d}  {elapsed:5.1f}s  {snippet}")

    return Panel(Text("\n".join(lines)), title=f"Active ({len(active)})", border_style="yellow")
```

**Step 4: Run tests to verify they pass**

Run: `cd projects/elenchus && uv run pytest tests/test_benchmark_display.py -v`
Expected: All tests pass

**Step 5: Lint and format**

Run: `cd projects/elenchus && uv run ruff check scripts/benchmark_display.py tests/test_benchmark_display.py && uv run ruff format scripts/benchmark_display.py tests/test_benchmark_display.py`

**Step 6: Commit**

```bash
git add scripts/benchmark_display.py tests/test_benchmark_display.py
git commit -m "feat(benchmark): add Rich display helpers with tests"
```

---

### Task 3: Create the live benchmark script

The main script. Uses `asyncio.as_completed()` for streaming, Rich `Live` for redraw, and a background task for timer ticks.

**Files:**
- Create: `scripts/benchmark_live.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Live benchmark dashboard — Rich terminal UI with streaming results.

Same pipeline as benchmark_probe.py (council → probe) but displays results
in a live-updating terminal dashboard as each problem completes.

Usage:
    uv run python scripts/benchmark_live.py --limit 20 --concurrency 2 --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console, Group
from rich.live import Live

from benchmark_display import ActiveProblem, make_active_panel, make_header, make_results_table
from benchmark_probe import (
    ProbeProblemResult,
    build_summary,
    load_calibration_problems,
    print_results,
    run_one_problem,
)
from elenchus.config import get_model_config


def build_layout(
    *,
    model_fast: str,
    model_capable: str,
    completed: int,
    total: int,
    correct: int,
    errors: int,
    elapsed: float,
    results: list[ProbeProblemResult],
    active: list[ActiveProblem],
    now: float,
) -> Group:
    """Compose the three-panel dashboard layout."""
    header = make_header(
        model_fast=model_fast, model_capable=model_capable,
        completed=completed, total=total, correct=correct,
        errors=errors, elapsed=elapsed,
    )
    table = make_results_table(results)
    active_panel = make_active_panel(active, now=now)
    return Group(header, table, active_panel)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Live benchmark dashboard")
    parser.add_argument("--limit", type=int, default=0, help="Max problems (0=all)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Relative tolerance")
    parser.add_argument("--output", type=str, default=None, help="Write JSON to file")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent problems")
    args = parser.parse_args()

    get_model_config.cache_clear()
    config = get_model_config()

    problems = load_calibration_problems()
    if args.category:
        problems = [p for p in problems if p["category"] == args.category]
    if args.limit:
        problems = problems[: args.limit]

    total = len(problems)
    semaphore = asyncio.Semaphore(args.concurrency)
    console = Console()

    # State
    results: list[ProbeProblemResult] = []
    active: list[ActiveProblem] = []
    correct = 0
    error_count = 0
    t_start = time.monotonic()

    # Create indexed tasks so we can track which problem each future belongs to
    async def run_indexed(idx: int, problem: dict) -> tuple[int, ProbeProblemResult]:
        return idx, await run_one_problem(problem, args.tolerance, semaphore)

    # Build futures
    futures = [
        asyncio.ensure_future(run_indexed(i, p))
        for i, p in enumerate(problems)
    ]

    # Track active problems
    for i, p in enumerate(problems):
        active.append(ActiveProblem(
            index=i + 1,
            question=p["question"][:50],
            start_time=time.monotonic(),
        ))

    # Background timer task — ticks active panel every second
    stop_timer = asyncio.Event()

    async def tick_timer(live: Live) -> None:
        while not stop_timer.is_set():
            now = time.monotonic()
            layout = build_layout(
                model_fast=config.fast, model_capable=config.capable,
                completed=len(results), total=total,
                correct=correct, errors=error_count,
                elapsed=now - t_start,
                results=results, active=active, now=now,
            )
            live.update(layout)
            try:
                await asyncio.wait_for(stop_timer.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    with Live(console=console, refresh_per_second=2, transient=False) as live:
        # Start timer
        timer_task = asyncio.create_task(tick_timer(live))

        # Stream results as they complete
        for coro in asyncio.as_completed(futures):
            idx, result = await coro

            # Update state
            results.append(result)
            if result.answer_correct:
                correct += 1
            if result.error:
                error_count += 1

            # Remove from active list
            active = [a for a in active if a.index != idx + 1]

            # Redraw
            now = time.monotonic()
            layout = build_layout(
                model_fast=config.fast, model_capable=config.capable,
                completed=len(results), total=total,
                correct=correct, errors=error_count,
                elapsed=now - t_start,
                results=results, active=active, now=now,
            )
            live.update(layout)

        # Stop timer
        stop_timer.set()
        await timer_task

    # Post-completion: static summary (same as benchmark_probe.py)
    wall_time = time.monotonic() - t_start
    summary = build_summary(results, config)
    summary.total_elapsed_s = round(wall_time, 1)
    print_results(results, summary)

    if args.output:
        output_data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        console.print(f"\n  Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Verify the script parses without errors**

Run: `cd projects/elenchus && uv run python -c "import sys; sys.path.insert(0, 'scripts'); import benchmark_live"`
Expected: No import errors

**Step 3: Lint and format**

Run: `cd projects/elenchus && uv run ruff check scripts/benchmark_live.py && uv run ruff format scripts/benchmark_live.py`

**Step 4: Commit**

```bash
git add scripts/benchmark_live.py
git commit -m "feat(benchmark): add live dashboard script with Rich UI"
```

---

### Task 4: Fix imports — benchmark_probe needs `load_calibration_problems` exported

`benchmark_live.py` imports `load_calibration_problems` from `benchmark_probe`. Check that `benchmark_probe.py` actually imports it at module level (it does — line 41). Also verify `print_results` and `build_summary` are importable. If any aren't at module scope, the import will fail.

**Files:**
- Possibly modify: `scripts/benchmark_live.py` (adjust imports if needed)

**Step 1: Test the cross-script import**

Run: `cd projects/elenchus && uv run python -c "import sys; sys.path.insert(0, 'scripts'); from benchmark_probe import run_one_problem, ProbeProblemResult, build_summary, print_results, load_calibration_problems; print('OK')"`
Expected: prints "OK"

If this fails, adjust `benchmark_live.py` to import `load_calibration_problems` directly from the elenchus package instead:

```python
from elenchus.calibration.dataset import load_calibration_problems
```

**Step 2: If changes were needed, commit**

```bash
git add scripts/benchmark_live.py
git commit -m "fix(benchmark): adjust live dashboard imports"
```

---

### Task 5: Run all existing tests to verify nothing is broken

**Step 1: Run the full test suite**

Run: `cd projects/elenchus && uv run pytest -x -q`
Expected: All existing tests pass (174+), plus new display tests

**Step 2: Run lint**

Run: `cd projects/elenchus && uv run ruff check src/ tests/ scripts/`
Expected: No errors

---

### Task 6: End-to-end smoke test

Run the live dashboard with a small limit to verify it works.

**Step 1: Run with --limit 3**

Run:
```bash
cd projects/elenchus && \
OPENAI_API_KEY=maei-local \
OPENAI_API_BASE=http://localhost:8317/v1 \
ELENCHUS_MODEL_FAST=openai/gemini-2.5-flash \
ELENCHUS_MODEL_CAPABLE=openai/gemini-2.5-flash \
uv run python scripts/benchmark_live.py --limit 3 --output /tmp/live_test.json
```

Expected:
- Live dashboard appears with header, empty table, 3 active problems
- Rows appear one by one as problems complete
- Active panel shrinks as problems finish
- Static summary prints after completion
- JSON written to `/tmp/live_test.json`

**Step 2: Verify JSON output matches expected format**

Run: `python -c "import json; d = json.load(open('/tmp/live_test.json')); print(f'Total: {d[\"summary\"][\"total\"]}, Results: {len(d[\"results\"])}')"`
Expected: `Total: 3, Results: 3`

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(benchmark): live dashboard verified end-to-end"
```

---

### Task 7: Push and verify CI

**Step 1: Push**

Run: `cd projects/elenchus && git push`

**Step 2: Verify CI passes**

Run: `gh pr checks` or check GitHub Actions
Expected: All checks green

---

## Notes

- `benchmark_probe.py` is NOT modified. The live script imports from it.
- Both scripts share the same CLI flags for consistency.
- Results appear in completion order (not problem order) since `as_completed` returns whichever finishes first. The `#` column shows the original problem index.
- The `transient=False` on Rich Live keeps the final dashboard state on screen after the Live context exits, then the static summary prints below it.
- If `rich` is not installed (e.g. production env without dev deps), the script will fail with a clear import error — this is expected since it's a dev tool.
