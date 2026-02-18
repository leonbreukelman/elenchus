#!/usr/bin/env python3
"""Benchmark the Deutsch Probe — tests explanation quality, not just accuracy.

Runs the FULL pipeline: council → probe (extract constraints, perturb,
instruct under perturbation, ground truth, score alignment, mechanism judge).

Reports:
  - Probe verdicts (verified / uncertain / refuted)
  - Alignment scores (instructed vs ground truth under perturbation)
  - Mechanism quality scores
  - Whether correct-answer problems get verified vs refuted explanations

Usage — CLIProxyAPI (Gemini, free):

    OPENAI_API_KEY=maei-local \\
    OPENAI_API_BASE=http://localhost:8317/v1 \\
    ELENCHUS_MODEL_FAST=openai/gemini-2.5-flash \\
    ELENCHUS_MODEL_CAPABLE=openai/gemini-2.5-pro \\
    uv run python scripts/benchmark_probe.py --limit 5

Options:
    --limit N         Run only the first N problems (default: all)
    --category CAT    Filter to a specific category
    --output FILE     Write JSON results to FILE
    --concurrency N   Max concurrent problems (default: 2)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from elenchus.calibration.dataset import load_calibration_problems
from elenchus.config import get_model_config
from elenchus.council.algebraic import AlgebraicCouncilor
from elenchus.council.consensus import evaluate_consensus
from elenchus.council.numerical import NumericalCouncilor
from elenchus.council.symbolic import SymbolicCouncilor
from elenchus.probe.graph import build_probe_graph
from elenchus.state import CouncilResult, CouncilorResult, RoutingResult
from elenchus.tools.sympy_tools import answers_match_numeric


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProbeProblemResult:
    """Result for one problem through the full pipeline."""

    question: str
    category: str
    source: str
    expected: float

    # Council results
    consensus_answer: float | None
    agreement: str
    answer_correct: bool

    # Probe results
    probe_verdict: str  # verified / uncertain / refuted / error
    probe_score: float
    sensitivity_map: dict[str, float]
    mechanism_scores: list[float]
    num_perturbations: int
    num_ground_truths_ok: int

    elapsed_s: float
    error: str | None = None


@dataclass
class ProbeBenchmarkSummary:
    model_fast: str
    model_capable: str
    total: int
    answer_correct: int
    answer_accuracy_pct: float

    # Probe verdicts
    verified: int
    uncertain: int
    refuted: int
    probe_errors: int

    # Cross-tabulation: answer correctness × probe verdict
    correct_and_verified: int
    correct_and_refuted: int
    wrong_and_verified: int
    wrong_and_refuted: int

    mean_probe_score: float
    mean_mechanism_score: float
    mean_latency_s: float
    total_elapsed_s: float

    by_category: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def run_one_problem(
    problem: dict,
    rel_tol: float,
    semaphore: asyncio.Semaphore,
) -> ProbeProblemResult:
    """Run one problem through council + Deutsch Probe."""
    question = problem["question"]
    expected = float(problem["expected_answer"])
    category = problem["category"]
    source = problem["source"]

    async with semaphore:
        t0 = time.monotonic()
        try:
            # --- Council phase ---
            councilors = [AlgebraicCouncilor(), NumericalCouncilor(), SymbolicCouncilor()]
            raw_results = await asyncio.gather(
                *[c.solve(question) for c in councilors],
                return_exceptions=True,
            )

            results: list[CouncilorResult] = []
            for r in raw_results:
                if not isinstance(r, Exception):
                    results.append(r)

            if not results:
                # Log the specific error from the first failed councilor for debugging
                for r in raw_results:
                    if isinstance(r, Exception):
                        print(f"Councilor failed: {r}", file=sys.stderr)
                return ProbeProblemResult(
                    question=question[:80], category=category, source=source,
                    expected=expected, consensus_answer=None, agreement="none",
                    answer_correct=False, probe_verdict="error", probe_score=0.0,
                    sensitivity_map={}, mechanism_scores=[], num_perturbations=0,
                    num_ground_truths_ok=0, elapsed_s=round(time.monotonic() - t0, 2),
                    error="All councilors failed",
                )

            consensus = evaluate_consensus(results, rel_tol=rel_tol)
            answer_correct = False
            if consensus.answer is not None:
                try:
                    answer_correct = answers_match_numeric(
                        float(consensus.answer), expected, rel_tol
                    )
                except (ValueError, TypeError):
                    pass

            # --- Probe phase ---
            council_result = CouncilResult(
                problem=question,
                domain="benchmark",
                routing=RoutingResult(
                    domain="benchmark",
                    problem_type="word_problem",
                    extracted_variables=[],
                    complexity="medium",
                ),
                consensus=consensus,
                councilor_results=results,
            )

            probe_graph = build_probe_graph()
            probe_state = await probe_graph.ainvoke({
                "council_result": council_result,
                "perturbation_budget": 3,
                "confidence_threshold": 0.80,
                "reject_threshold": 0.50,
            })

            probe_result = probe_state.get("probe_result")
            elapsed = time.monotonic() - t0

            if probe_result is None:
                return ProbeProblemResult(
                    question=question[:80], category=category, source=source,
                    expected=expected,
                    consensus_answer=consensus.answer,
                    agreement=consensus.agreement,
                    answer_correct=answer_correct,
                    probe_verdict="error", probe_score=0.0,
                    sensitivity_map={}, mechanism_scores=[],
                    num_perturbations=len(probe_state.get("perturbations", [])),
                    num_ground_truths_ok=0,
                    elapsed_s=round(elapsed, 2),
                    error="Probe produced no result",
                )

            mechanism_scores = [
                sr.reasoning_quality
                for sr in probe_result.results
                if sr.reasoning_quality is not None
            ]

            return ProbeProblemResult(
                question=question[:80],
                category=category,
                source=source,
                expected=expected,
                consensus_answer=consensus.answer,
                agreement=consensus.agreement,
                answer_correct=answer_correct,
                probe_verdict=probe_result.verdict.value,
                probe_score=round(probe_result.overall_score, 3),
                sensitivity_map={k: round(v, 3) for k, v in probe_result.sensitivity_map.items()},
                mechanism_scores=[round(s, 3) for s in mechanism_scores],
                num_perturbations=len(probe_state.get("perturbations", [])),
                num_ground_truths_ok=sum(
                    1 for gt in probe_state.get("ground_truths", []) if gt.get("success")
                ),
                elapsed_s=round(elapsed, 2),
            )

        except Exception as exc:
            return ProbeProblemResult(
                question=question[:80], category=category, source=source,
                expected=expected, consensus_answer=None, agreement="none",
                answer_correct=False, probe_verdict="error", probe_score=0.0,
                sensitivity_map={}, mechanism_scores=[], num_perturbations=0,
                num_ground_truths_ok=0, elapsed_s=round(time.monotonic() - t0, 2),
                error=str(exc),
            )


def build_summary(
    results: list[ProbeProblemResult], config
) -> ProbeBenchmarkSummary:
    total = len(results)
    correct = sum(1 for r in results if r.answer_correct)
    verified = sum(1 for r in results if r.probe_verdict == "verified")
    uncertain = sum(1 for r in results if r.probe_verdict == "uncertain")
    refuted = sum(1 for r in results if r.probe_verdict == "refuted")
    errors = sum(1 for r in results if r.probe_verdict == "error")

    # Cross-tabulation
    cv = sum(1 for r in results if r.answer_correct and r.probe_verdict == "verified")
    cr = sum(1 for r in results if r.answer_correct and r.probe_verdict == "refuted")
    wv = sum(1 for r in results if not r.answer_correct and r.probe_verdict == "verified")
    wr = sum(1 for r in results if not r.answer_correct and r.probe_verdict == "refuted")

    scores = [r.probe_score for r in results if r.probe_verdict != "error"]
    mech = [s for r in results for s in r.mechanism_scores]
    latencies = [r.elapsed_s for r in results]

    # Per-category
    cats: dict[str, list[ProbeProblemResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    by_category = {}
    for cat, cat_results in sorted(cats.items()):
        n = len(cat_results)
        cat_correct = sum(1 for r in cat_results if r.answer_correct)
        cat_verified = sum(1 for r in cat_results if r.probe_verdict == "verified")
        cat_scores = [r.probe_score for r in cat_results if r.probe_verdict != "error"]
        by_category[cat] = {
            "total": n,
            "correct": cat_correct,
            "accuracy_pct": round(cat_correct / n * 100, 1) if n else 0,
            "verified": cat_verified,
            "verified_pct": round(cat_verified / n * 100, 1) if n else 0,
            "mean_probe_score": round(sum(cat_scores) / len(cat_scores), 3) if cat_scores else 0,
        }

    return ProbeBenchmarkSummary(
        model_fast=config.fast,
        model_capable=config.capable,
        total=total,
        answer_correct=correct,
        answer_accuracy_pct=round(correct / total * 100, 1) if total else 0,
        verified=verified,
        uncertain=uncertain,
        refuted=refuted,
        probe_errors=errors,
        correct_and_verified=cv,
        correct_and_refuted=cr,
        wrong_and_verified=wv,
        wrong_and_refuted=wr,
        mean_probe_score=round(sum(scores) / len(scores), 3) if scores else 0,
        mean_mechanism_score=round(sum(mech) / len(mech), 3) if mech else 0,
        mean_latency_s=round(sum(latencies) / total, 2) if total else 0,
        total_elapsed_s=round(sum(latencies), 1),
        by_category=by_category,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[ProbeProblemResult], summary: ProbeBenchmarkSummary) -> None:
    print("\n" + "=" * 80)
    print("  DEUTSCH PROBE BENCHMARK — Explanation Quality")
    print("=" * 80)
    print(f"  Models:  fast={summary.model_fast}")
    print(f"           capable={summary.model_capable}")
    print(f"  Problems: {summary.total}")
    print()

    # Per-problem table
    hdr = f"  {'#':>3}  {'Ans':>3}  {'Verdict':>10}  {'Score':>5}  {'Mech':>5}  {'Pert':>4}  {'Time':>6}  Question"
    print(hdr)
    print(f"  {'─'*3}  {'─'*3}  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*4}  {'─'*6}  {'─'*40}")

    for i, r in enumerate(results, 1):
        ans_mark = " ✓" if r.answer_correct else " ✗"
        if r.error:
            ans_mark = "ERR"

        verdict_display = r.probe_verdict.upper() if r.probe_verdict == "verified" else r.probe_verdict
        score = f"{r.probe_score:.2f}" if r.probe_score else "  — "
        mech_avg = "  — "
        if r.mechanism_scores:
            mech_avg = f"{sum(r.mechanism_scores)/len(r.mechanism_scores):.2f}"
        perts = f"{r.num_ground_truths_ok}/{r.num_perturbations}"

        print(f"  {i:3d}  {ans_mark:>3}  {verdict_display:>10}  {score:>5}  {mech_avg:>5}  {perts:>4}  {r.elapsed_s:5.1f}s  {r.question[:40]}")

    # Summary
    print()
    print("  " + "─" * 78)
    print(f"  Answer accuracy:     {summary.answer_correct}/{summary.total} ({summary.answer_accuracy_pct}%)")
    print()
    print(f"  Probe verdicts:      verified={summary.verified}  uncertain={summary.uncertain}  "
          f"refuted={summary.refuted}  errors={summary.probe_errors}")
    print(f"  Mean probe score:    {summary.mean_probe_score:.3f}")
    print(f"  Mean mechanism score:{summary.mean_mechanism_score:.3f}")
    print()

    # Cross-tabulation — the key insight
    print("  ┌─────────────────────────┬───────────┬───────────┐")
    print("  │                         │  Verified │  Refuted  │")
    print("  ├─────────────────────────┼───────────┼───────────┤")
    print(f"  │  Answer CORRECT         │    {summary.correct_and_verified:3d}    │    {summary.correct_and_refuted:3d}    │")
    print(f"  │  Answer WRONG           │    {summary.wrong_and_verified:3d}    │    {summary.wrong_and_refuted:3d}    │")
    print("  └─────────────────────────┴───────────┴───────────┘")
    print()

    if summary.wrong_and_verified:
        print("  ⚠  WRONG + VERIFIED: model got wrong answer but probe thinks reasoning is sound")
    if summary.correct_and_refuted:
        print("  ⚠  CORRECT + REFUTED: model got right answer but explanation is easy to vary")

    print(f"\n  Latency:  mean={summary.mean_latency_s}s  total={summary.total_elapsed_s}s")

    # Per-category
    print()
    print(f"  {'Category':<20}  {'Accuracy':>8}  {'Verified':>8}  {'ProbeScore':>10}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*10}")
    for cat, s in sorted(summary.by_category.items()):
        print(f"  {cat:<20}  {s['accuracy_pct']:>7.1f}%  {s['verified_pct']:>7.1f}%  {s['mean_probe_score']:>10.3f}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Deutsch Probe explanation quality")
    parser.add_argument("--limit", type=int, default=0, help="Max problems (0=all)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Relative tolerance")
    parser.add_argument("--output", type=str, default=None, help="Write JSON to file")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent problems")
    args = parser.parse_args()

    get_model_config.cache_clear()
    config = get_model_config()

    print(f"\n  ► Models: fast={config.fast}  capable={config.capable}")

    problems = load_calibration_problems()
    if args.category:
        problems = [p for p in problems if p["category"] == args.category]
    if args.limit:
        problems = problems[:args.limit]

    print(f"  ► Running {len(problems)} problems through full pipeline (council + probe)")
    print(f"  ► Concurrency={args.concurrency}, tolerance={args.tolerance}")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)

    t_start = time.monotonic()
    results = await asyncio.gather(
        *[run_one_problem(p, args.tolerance, semaphore) for p in problems]
    )
    wall_time = time.monotonic() - t_start

    summary = build_summary(list(results), config)
    summary.total_elapsed_s = round(wall_time, 1)

    print_results(list(results), summary)

    if args.output:
        output_data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\n  Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
