#!/usr/bin/env python
"""Evaluation harness — runs the Elenchus pipeline against official math benchmark test splits.

Pulls problems from GSM8K (test) and MATH (test), excludes any overlap with the
calibration dataset, runs each through the full engine graph, and reports accuracy
broken down by source, category, and probe verdict.

Usage:
    uv run python scripts/evaluate.py --count 100 --concurrency 5 --seed 42
    uv run python scripts/evaluate.py --count 20  # quick sanity check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# GSM8K helpers (copied from build_calibration_dataset.py)
# ---------------------------------------------------------------------------


def _extract_gsm8k_answer(answer_text: str) -> float | None:
    """Extract the numeric answer after #### in GSM8K answer field."""
    m = re.search(r"####\s*([+-]?[\d,]+(?:\.\d+)?)", answer_text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


_GSM8K_RATE_KEYWORDS = [
    "per hour",
    "per minute",
    "per day",
    "per week",
    "per month",
    "miles per",
    "km per",
    "gallons per",
    "liters per",
    "speed",
    "rate",
    "faster",
    "slower",
    "fill",
    "drain",
    "empty",
    "pump",
    "working together",
    "work together",
    "distance",
    "travel",
    "drove",
    "walked",
    "ran",
    "biked",
]

_GSM8K_PERCENTAGE_KEYWORDS = [
    "percent",
    "%",
    "discount",
    "markup",
    "tax",
    "tip",
    "increase by",
    "decrease by",
    "profit margin",
]

_GSM8K_PROPORTION_KEYWORDS = [
    "ratio",
    "proportion",
    "divided among",
    "split",
    "shared",
    "scale",
    "recipe",
    "mixture",
]

_GSM8K_SKIP_KEYWORDS = [
    "how many",
    "count",
    "total number of",
]


def _categorize_gsm8k(question: str) -> str:
    """Categorize a GSM8K problem using keyword heuristics."""
    q = question.lower()
    if any(kw in q for kw in _GSM8K_RATE_KEYWORDS):
        return "rate"
    if any(kw in q for kw in _GSM8K_PERCENTAGE_KEYWORDS):
        return "percentage"
    if any(kw in q for kw in _GSM8K_PROPORTION_KEYWORDS):
        return "proportion"
    return "arithmetic"


def _is_interesting_gsm8k(question: str) -> bool:
    """Prefer rate/work/distance/fill problems; skip trivial counting."""
    q = question.lower()
    if any(kw in q for kw in _GSM8K_SKIP_KEYWORDS):
        if any(kw in q for kw in _GSM8K_RATE_KEYWORDS):
            return True
        return False
    return True


# ---------------------------------------------------------------------------
# MATH helpers (copied from build_calibration_dataset.py)
# ---------------------------------------------------------------------------


_MATH_EQUATION_KEYWORDS = [
    "solve",
    "find the value",
    "what is the value",
    "root",
    "solution",
]

_MATH_SYSTEM_KEYWORDS = [
    "system",
    "simultaneously",
    "two equations",
    "and",
    "if.*and",
]

_MATH_POLYNOMIAL_KEYWORDS = [
    "polynomial",
    "degree",
    "coefficient",
    "factor",
    "quadratic",
    "cubic",
]

_MATH_INEQUALITY_KEYWORDS = [
    "inequality",
    "greater than",
    "less than",
    "at least",
    "at most",
    "minimum",
    "maximum",
]


def _extract_math_answer(solution: str) -> float | None:
    r"""Extract numeric answer from \boxed{...} in MATH solution field."""
    m = re.search(r"\\boxed\{([^}]+)\}", solution)
    if not m:
        return None
    content = m.group(1).strip()
    content = content.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", content):
        try:
            return float(content)
        except ValueError:
            return None
    return None


def _categorize_math(problem: str) -> str:
    """Categorize a MATH problem."""
    q = problem.lower()
    if any(kw in q for kw in _MATH_INEQUALITY_KEYWORDS):
        return "inequality"
    if any(kw in q for kw in _MATH_POLYNOMIAL_KEYWORDS):
        return "polynomial"
    if any(kw in q for kw in _MATH_SYSTEM_KEYWORDS):
        return "system"
    if any(kw in q for kw in _MATH_EQUATION_KEYWORDS):
        return "equation"
    return "equation"


# ---------------------------------------------------------------------------
# Data sourcing
# ---------------------------------------------------------------------------


def _build_exclusion_set() -> set[str]:
    """Build the set of source_ids already in the calibration dataset."""
    from elenchus.calibration.dataset import load_calibration_problems

    return {p["source_id"] for p in load_calibration_problems()}


def pull_gsm8k_test(target: int, exclusion_ids: set[str], seed: int) -> list[dict]:
    """Pull problems from GSM8K test split, excluding calibration overlaps."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test")

    candidates: list[dict] = []
    for i, row in enumerate(ds):
        source_id = f"gsm8k-test-{i}"
        if source_id in exclusion_ids:
            continue
        answer = _extract_gsm8k_answer(row["answer"])
        if answer is None:
            continue
        question = row["question"].strip()
        if not _is_interesting_gsm8k(question):
            continue
        category = _categorize_gsm8k(question)
        candidates.append(
            {
                "question": question,
                "expected_answer": answer,
                "category": category,
                "source": "gsm8k",
                "source_id": source_id,
            }
        )

    rng.shuffle(candidates)
    return candidates[:target]


def _extract_math500_answer(answer_str: str) -> float | None:
    """Extract a plain numeric answer from MATH-500 answer field."""
    cleaned = answer_str.strip().replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def pull_math_test(target: int, exclusion_ids: set[str], seed: int) -> list[dict]:
    """Pull problems from MATH-500 test split (Algebra/Prealgebra Level 1-3), excluding calibration overlaps."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    candidates: list[dict] = []
    for row in ds:
        unique_id = row.get("unique_id", "")
        source_id = f"math500-{unique_id}"
        if source_id in exclusion_ids:
            continue
        subject = row.get("subject", "")
        if subject not in ("Algebra", "Prealgebra"):
            continue
        level = row.get("level", 99)
        if level > 3:
            continue
        answer = _extract_math500_answer(row.get("answer", ""))
        if answer is None:
            continue
        question = row["problem"].strip()
        category = _categorize_math(question)
        candidates.append(
            {
                "question": question,
                "expected_answer": answer,
                "category": category,
                "source": "math500",
                "source_id": source_id,
            }
        )

    rng.shuffle(candidates)
    return candidates[:target]


def load_eval_problems(count: int, gsm8k_ratio: float, seed: int) -> list[dict]:
    """Load evaluation problems from test splits, excluding calibration data."""
    exclusion_ids = _build_exclusion_set()
    logger.info("built_exclusion_set", size=len(exclusion_ids))

    gsm8k_target = round(count * gsm8k_ratio)
    math_target = count - gsm8k_target

    logger.info("loading_gsm8k", target=gsm8k_target)
    gsm8k = pull_gsm8k_test(gsm8k_target, exclusion_ids, seed)
    logger.info("loaded_gsm8k", count=len(gsm8k))

    logger.info("loading_math", target=math_target)
    math = pull_math_test(math_target, exclusion_ids, seed)
    logger.info("loaded_math", count=len(math))

    combined = gsm8k + math
    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _is_correct(actual: float, expected: float, rel_tol: float = 0.01) -> bool:
    """Check if actual answer is within relative tolerance of expected."""
    return abs(actual - expected) / max(abs(expected), 1e-10) <= rel_tol


async def run_single(
    problem: dict,
    graph: Any,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single problem through the engine, capturing results and timing."""
    result_dict: dict[str, Any] = {
        "question": problem["question"],
        "expected_answer": problem["expected_answer"],
        "category": problem["category"],
        "source": problem["source"],
        "source_id": problem["source_id"],
        "actual_answer": None,
        "correct": False,
        "probe_verdict": None,
        "probe_score": None,
        "sensitivity_map": None,
        "confidence": None,
        "wall_time_seconds": None,
        "error": None,
    }

    async with semaphore:
        t0 = time.monotonic()
        try:
            state = await graph.ainvoke({"problem": problem["question"]})
            elapsed = time.monotonic() - t0
            result_dict["wall_time_seconds"] = round(elapsed, 2)

            vr = state.get("verified_result")
            if vr is None:
                result_dict["error"] = "No verified_result in engine output"
                return result_dict

            actual = vr.answer
            try:
                actual_float = float(actual)
            except (TypeError, ValueError):
                actual_float = None

            result_dict["actual_answer"] = actual_float
            result_dict["confidence"] = vr.confidence

            if actual_float is not None:
                result_dict["correct"] = _is_correct(actual_float, problem["expected_answer"])

            if vr.probe_verdict is not None:
                result_dict["probe_verdict"] = vr.probe_verdict.value
            result_dict["probe_score"] = vr.explanation_quality
            result_dict["sensitivity_map"] = vr.sensitivity_map

        except Exception as exc:
            elapsed = time.monotonic() - t0
            result_dict["wall_time_seconds"] = round(elapsed, 2)
            result_dict["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("pipeline_error", source_id=problem["source_id"], error=str(exc))

    return result_dict


async def run_evaluation(
    problems: list[dict],
    concurrency: int,
) -> list[dict]:
    """Run all problems through the pipeline with bounded concurrency."""
    from elenchus.engine import build_engine_graph

    graph = build_engine_graph()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [run_single(p, graph, semaphore) for p in problems]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format seconds as Xm Ys."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def print_summary(results: list[dict], total_elapsed: float) -> None:
    """Print a human-readable summary of evaluation results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r["error"] is not None)
    wall_times = [r["wall_time_seconds"] for r in results if r["wall_time_seconds"] is not None]
    avg_wall = sum(wall_times) / len(wall_times) if wall_times else 0.0
    pct = (correct / total * 100) if total else 0.0

    print("\n=== Elenchus Evaluation ===")
    print(f"Problems: {total} | Correct: {correct} ({pct:.1f}%) | Errors: {errors}")
    print(f"Wall time: {_format_duration(total_elapsed)} | Avg per problem: {avg_wall:.1f}s")

    # By source
    sources: dict[str, list[dict]] = {}
    for r in results:
        sources.setdefault(r["source"], []).append(r)

    print("\nBy source:")
    for source in sorted(sources):
        group = sources[source]
        src_correct = sum(1 for r in group if r["correct"])
        src_total = len(group)
        src_pct = (src_correct / src_total * 100) if src_total else 0.0
        print(f"  {source:12s} {src_correct}/{src_total} ({src_pct:.1f}%)")

    # By category
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print("\nBy category:")
    for cat in sorted(categories):
        group = categories[cat]
        cat_correct = sum(1 for r in group if r["correct"])
        cat_total = len(group)
        cat_pct = (cat_correct / cat_total * 100) if cat_total else 0.0
        print(f"  {cat:20s} {cat_correct}/{cat_total} ({cat_pct:.1f}%)")

    # Probe verdicts — separate correct vs wrong
    correct_results = [r for r in results if r["correct"] and r["probe_verdict"] is not None]
    wrong_results = [r for r in results if not r["correct"] and r["error"] is None and r["probe_verdict"] is not None]

    if correct_results:
        print("\nProbe verdicts (correct answers only):")
        verdict_counts: dict[str, int] = {}
        for r in correct_results:
            v = r["probe_verdict"]
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        verdict_total = sum(verdict_counts.values())
        for verdict in ["hard_to_vary", "partially_coupled", "easy_to_vary"]:
            n = verdict_counts.get(verdict, 0)
            pct_v = (n / verdict_total * 100) if verdict_total else 0.0
            print(f"  {verdict:22s} {n:4d} ({pct_v:.1f}%)")

    if wrong_results:
        print("\nProbe verdicts (wrong answers):")
        verdict_counts_wrong: dict[str, int] = {}
        for r in wrong_results:
            v = r["probe_verdict"]
            verdict_counts_wrong[v] = verdict_counts_wrong.get(v, 0) + 1
        verdict_total_wrong = sum(verdict_counts_wrong.values())
        for verdict in ["hard_to_vary", "partially_coupled", "easy_to_vary"]:
            n = verdict_counts_wrong.get(verdict, 0)
            pct_v = (n / verdict_total_wrong * 100) if verdict_total_wrong else 0.0
            print(f"  {verdict:22s} {n:4d} ({pct_v:.1f}%)")

    # Average probe scores
    correct_scores = [r["probe_score"] for r in results if r["correct"] and r["probe_score"] is not None]
    wrong_scores = [
        r["probe_score"] for r in results if not r["correct"] and r["error"] is None and r["probe_score"] is not None
    ]
    avg_correct = sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
    avg_wrong = sum(wrong_scores) / len(wrong_scores) if wrong_scores else 0.0
    print(f"\nAvg probe score: correct={avg_correct:.2f}, wrong={avg_wrong:.2f}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(results: list[dict]) -> Path:
    """Save results to a timestamped JSON file in results/."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    output_path = results_dir / f"eval-{timestamp}.json"

    # Convert any non-serializable values
    serializable = []
    for r in results:
        entry = dict(r)
        # sensitivity_map values are already floats; just ensure serializability
        if entry.get("sensitivity_map") is not None:
            entry["sensitivity_map"] = {str(k): float(v) for k, v in entry["sensitivity_map"].items()}
        serializable.append(entry)

    output_path.write_text(json.dumps(serializable, indent=2))
    logger.info("results_saved", path=str(output_path), count=len(results))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Elenchus evaluation against benchmark test splits")
    parser.add_argument("--count", type=int, default=100, help="Total problems to evaluate (default: 100)")
    parser.add_argument("--concurrency", type=int, default=5, help="Max parallel pipeline runs (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--gsm8k-ratio", type=float, default=0.6, help="Fraction of problems from GSM8K (default: 0.6)")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    """Async entry point."""
    # Validate API key is available
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set — cannot run pipeline")
        raise SystemExit(1)

    logger.info(
        "evaluation_start",
        count=args.count,
        concurrency=args.concurrency,
        seed=args.seed,
        gsm8k_ratio=args.gsm8k_ratio,
    )

    problems = load_eval_problems(args.count, args.gsm8k_ratio, args.seed)
    if not problems:
        logger.error("no_problems_loaded")
        raise SystemExit(1)

    logger.info("problems_loaded", count=len(problems))

    t0 = time.monotonic()
    results = await run_evaluation(problems, args.concurrency)
    elapsed = time.monotonic() - t0
    logger.info("evaluation_complete", wall_time=round(elapsed, 1))

    output_path = save_results(results)
    print_summary(results, elapsed)
    print(f"\nFull results: {output_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
