"""Rich display helpers for the live benchmark dashboard.

Pure rendering functions — no I/O, no async. Each returns a Rich
renderable (Panel, Table) that the Live context composites.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark_probe import ProbeProblemResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
        f"Progress: {completed}/{total}    Correct: {correct} ({pct})    Errors: {errors}    Elapsed: {elapsed:.0f}s",
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
            str(i),
            ans,
            verdict,
            score,
            mech,
            perts,
            f"{r.elapsed_s:.1f}s",
            r.question[:50],
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
