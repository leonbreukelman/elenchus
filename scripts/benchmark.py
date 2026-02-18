#!/usr/bin/env python3
"""Benchmark Elenchus accuracy across model combinations.

Runs the calibration dataset through the council (3 councilors → consensus)
and compares the consensus answer against expected answers.

Usage — CLIProxyAPI (Gemini, free):

    OPENAI_API_KEY=maei-local \\
    OPENAI_API_BASE=http://localhost:8317/v1 \\
    ELENCHUS_MODEL_FAST=openai/gemini-2.5-flash \\
    ELENCHUS_MODEL_CAPABLE=openai/gemini-2.5-pro \\
    uv run python scripts/benchmark.py

Usage — OpenRouter:

    OPENROUTER_API_KEY=sk-or-... \\
    ELENCHUS_MODEL_FAST=openrouter/google/gemini-2.0-flash-001 \\
    ELENCHUS_MODEL_CAPABLE=openrouter/anthropic/claude-sonnet-4-5-20250929 \\
    uv run python scripts/benchmark.py

Options:
    --limit N         Run only the first N problems (default: all 85)
    --category CAT    Filter to a specific category
    --tolerance TOL   Relative tolerance for answer matching (default: 0.01)
    --output FILE     Write JSON results to FILE
    --concurrency N   Max concurrent problems (default: 3)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure elenchus is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from elenchus.calibration.dataset import load_calibration_problems
from elenchus.config import get_model_config
from elenchus.council.algebraic import AlgebraicCouncilor
from elenchus.council.consensus import evaluate_consensus
from elenchus.council.numerical import NumericalCouncilor
from elenchus.council.symbolic import SymbolicCouncilor
from elenchus.tools.sympy_tools import answers_match_numeric


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProblemResult:
    question: str
    category: str
    source: str
    expected: float
    consensus_answer: float | None
    agreement: str
    confidence: float
    correct: bool
    councilor_answers: dict[str, float | None]
    elapsed_s: float
    error: str | None = None


@dataclass
class BenchmarkSummary:
    model_fast: str
    model_capable: str
    total: int
    correct: int
    accuracy_pct: float
    unanimous_pct: float
    majority_pct: float
    no_agreement_pct: float
    mean_latency_s: float
    median_latency_s: float
    total_elapsed_s: float
    by_category: dict[str, dict] = field(default_factory=dict)
    errors: int = 0


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def solve_one(
    problem: dict,
    rel_tol: float,
    semaphore: asyncio.Semaphore,
) -> ProblemResult:
    """Run one problem through the council and evaluate."""
    question = problem["question"]
    expected = float(problem["expected_answer"])
    category = problem["category"]
    source = problem["source"]

    async with semaphore:
        t0 = time.monotonic()
        try:
            councilors = [AlgebraicCouncilor(), NumericalCouncilor(), SymbolicCouncilor()]
            raw_results = await asyncio.gather(
                *[c.solve(question) for c in councilors],
                return_exceptions=True,
            )

            # Filter successful results
            results = []
            councilor_answers: dict[str, float | None] = {}
            for r, c in zip(raw_results, councilors):
                if isinstance(r, Exception):
                    councilor_answers[c.strategy] = None
                else:
                    results.append(r)
                    councilor_answers[c.strategy] = r.answer

            if not results:
                elapsed = time.monotonic() - t0
                return ProblemResult(
                    question=question[:80],
                    category=category,
                    source=source,
                    expected=expected,
                    consensus_answer=None,
                    agreement="none",
                    confidence=0.0,
                    correct=False,
                    councilor_answers=councilor_answers,
                    elapsed_s=round(elapsed, 2),
                    error="All councilors failed",
                )

            consensus = evaluate_consensus(results, rel_tol=rel_tol)
            elapsed = time.monotonic() - t0

            # Check correctness
            correct = False
            if consensus.answer is not None:
                try:
                    correct = answers_match_numeric(
                        float(consensus.answer), expected, rel_tol
                    )
                except (ValueError, TypeError):
                    correct = False

            return ProblemResult(
                question=question[:80],
                category=category,
                source=source,
                expected=expected,
                consensus_answer=consensus.answer,
                agreement=consensus.agreement,
                confidence=round(consensus.confidence, 3),
                correct=correct,
                councilor_answers=councilor_answers,
                elapsed_s=round(elapsed, 2),
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            return ProblemResult(
                question=question[:80],
                category=category,
                source=source,
                expected=expected,
                consensus_answer=None,
                agreement="none",
                confidence=0.0,
                correct=False,
                councilor_answers={},
                elapsed_s=round(elapsed, 2),
                error=str(exc),
            )


def build_summary(results: list[ProblemResult], config) -> BenchmarkSummary:
    """Aggregate per-problem results into a summary."""
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.error)
    unanimous = sum(1 for r in results if r.agreement == "unanimous")
    majority = sum(1 for r in results if r.agreement == "majority")
    no_agree = sum(1 for r in results if r.agreement == "none")
    latencies = [r.elapsed_s for r in results]

    sorted_lat = sorted(latencies)
    median = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0

    # Per-category breakdown
    cats: dict[str, list[ProblemResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    by_category = {}
    for cat, cat_results in sorted(cats.items()):
        cat_total = len(cat_results)
        cat_correct = sum(1 for r in cat_results if r.correct)
        by_category[cat] = {
            "total": cat_total,
            "correct": cat_correct,
            "accuracy_pct": round(cat_correct / cat_total * 100, 1) if cat_total else 0,
        }

    return BenchmarkSummary(
        model_fast=config.fast,
        model_capable=config.capable,
        total=total,
        correct=correct,
        accuracy_pct=round(correct / total * 100, 1) if total else 0,
        unanimous_pct=round(unanimous / total * 100, 1) if total else 0,
        majority_pct=round(majority / total * 100, 1) if total else 0,
        no_agreement_pct=round(no_agree / total * 100, 1) if total else 0,
        mean_latency_s=round(sum(latencies) / total, 2) if total else 0,
        median_latency_s=round(median, 2),
        total_elapsed_s=round(sum(latencies), 1),
        by_category=by_category,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results: list[ProblemResult], summary: BenchmarkSummary) -> None:
    """Print a formatted benchmark report to stdout."""
    print("\n" + "=" * 72)
    print("  ELENCHUS BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  Models:  fast={summary.model_fast}")
    print(f"           capable={summary.model_capable}")
    print(f"  Problems: {summary.total}")
    print()

    # Per-problem table
    print(f"  {'#':>3}  {'Correct':>7}  {'Agreement':>10}  {'Conf':>5}  {'Time':>6}  Question")
    print(f"  {'─'*3}  {'─'*7}  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*40}")

    for i, r in enumerate(results, 1):
        mark = "  ✓" if r.correct else "  ✗"
        if r.error:
            mark = " ERR"
        conf = f"{r.confidence:.2f}" if r.confidence else "  — "
        print(f"  {i:3d}  {mark:>7}  {r.agreement:>10}  {conf:>5}  {r.elapsed_s:5.1f}s  {r.question[:40]}")

    # Summary
    print()
    print("  " + "─" * 70)
    print(f"  Accuracy:      {summary.correct}/{summary.total} ({summary.accuracy_pct}%)")
    print(f"  Agreement:     unanimous={summary.unanimous_pct}%  "
          f"majority={summary.majority_pct}%  none={summary.no_agreement_pct}%")
    print(f"  Latency:       mean={summary.mean_latency_s}s  median={summary.median_latency_s}s")
    print(f"  Total time:    {summary.total_elapsed_s}s")
    if summary.errors:
        print(f"  Errors:        {summary.errors}")

    # Per-category
    print()
    print(f"  {'Category':<20}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*5}  {'─'*8}")
    for cat, stats in sorted(summary.by_category.items()):
        print(f"  {cat:<20}  {stats['correct']:>7}  {stats['total']:>5}  {stats['accuracy_pct']:>7.1f}%")

    print("=" * 72)

    # Show failures
    failures = [r for r in results if not r.correct]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):\n")
        for r in failures:
            exp = f"{r.expected}"
            got = f"{r.consensus_answer}" if r.consensus_answer is not None else "None"
            err = f" [{r.error}]" if r.error else ""
            print(f"    • {r.question[:60]}")
            print(f"      expected={exp}  got={got}  "
                  f"councilors={r.councilor_answers}{err}")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Elenchus council accuracy")
    parser.add_argument("--limit", type=int, default=0, help="Max problems to run (0=all)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Relative tolerance")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to file")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent problems")
    args = parser.parse_args()

    # Clear config cache so env vars take effect
    get_model_config.cache_clear()
    config = get_model_config()

    print(f"\n  ► Models: fast={config.fast}  capable={config.capable}")

    # Load problems
    problems = load_calibration_problems()
    if args.category:
        problems = [p for p in problems if p["category"] == args.category]
    if args.limit:
        problems = problems[:args.limit]

    print(f"  ► Running {len(problems)} problems (concurrency={args.concurrency}, tol={args.tolerance})")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)

    # Run benchmark
    t_start = time.monotonic()
    results = await asyncio.gather(
        *[solve_one(p, args.tolerance, semaphore) for p in problems]
    )
    wall_time = time.monotonic() - t_start

    summary = build_summary(list(results), config)
    summary.total_elapsed_s = round(wall_time, 1)

    print_results(list(results), summary)

    # Optionally write JSON
    if args.output:
        output_data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\n  Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
