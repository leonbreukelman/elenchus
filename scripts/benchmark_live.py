#!/usr/bin/env python3
"""Live benchmark dashboard — Rich terminal UI with streaming results.

Same pipeline as benchmark_probe.py (council + probe) but displays results
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

from benchmark_display import ActiveProblem, make_active_panel, make_header, make_results_table
from benchmark_probe import (
    ProbeProblemResult,
    build_summary,
    print_results,
    run_one_problem,
)
from rich.console import Console, Group
from rich.live import Live

from elenchus.calibration.dataset import load_calibration_problems
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
        model_fast=model_fast,
        model_capable=model_capable,
        completed=completed,
        total=total,
        correct=correct,
        errors=errors,
        elapsed=elapsed,
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
    futures = [asyncio.ensure_future(run_indexed(i, p)) for i, p in enumerate(problems)]

    # Track active problems
    for i, p in enumerate(problems):
        active.append(
            ActiveProblem(
                index=i + 1,
                question=p["question"][:50],
                start_time=time.monotonic(),
            )
        )

    # Background timer task — ticks active panel every second
    stop_timer = asyncio.Event()

    async def tick_timer(live: Live) -> None:
        while not stop_timer.is_set():
            now = time.monotonic()
            layout = build_layout(
                model_fast=config.fast,
                model_capable=config.capable,
                completed=len(results),
                total=total,
                correct=correct,
                errors=error_count,
                elapsed=now - t_start,
                results=results,
                active=active,
                now=now,
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
                model_fast=config.fast,
                model_capable=config.capable,
                completed=len(results),
                total=total,
                correct=correct,
                errors=error_count,
                elapsed=now - t_start,
                results=results,
                active=active,
                now=now,
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
