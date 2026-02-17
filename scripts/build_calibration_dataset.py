#!/usr/bin/env python
"""Build the calibration dataset from GSM8K, MATH, and hand-curated problems.

Pulls ~40 problems from GSM8K (openai/gsm8k, train split) and ~30 from MATH
(qwedsacf/competition_math, train split), filters for single numeric answers,
categorizes using keyword heuristics, then combines with ~15 hand-curated
physics/finance problems.  Outputs a Python module that replaces dataset.py.

Usage:
    uv run python scripts/build_calibration_dataset.py
"""

from __future__ import annotations

import random
import re
import textwrap
from pathlib import Path

from datasets import load_dataset

SEED = 42
random.seed(SEED)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "src" / "elenchus" / "calibration" / "dataset.py"


# ---------------------------------------------------------------------------
# GSM8K helpers
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
    # Skip purely counting problems
    if any(kw in q for kw in _GSM8K_SKIP_KEYWORDS):
        # But keep it if it also has rate/distance keywords
        if any(kw in q for kw in _GSM8K_RATE_KEYWORDS):
            return True
        return False
    return True


def pull_gsm8k(target: int = 40) -> list[dict]:
    """Pull and filter GSM8K problems."""
    ds = load_dataset("openai/gsm8k", "main", split="train")

    candidates = []
    for i, row in enumerate(ds):
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
                "source_id": f"gsm8k-train-{i}",
            }
        )

    # Prefer rate problems, then percentage, then proportion, then arithmetic
    priority = {"rate": 0, "percentage": 1, "proportion": 2, "arithmetic": 3}
    candidates.sort(key=lambda p: priority.get(p["category"], 99))

    # Sample ensuring category diversity
    by_cat: dict[str, list[dict]] = {}
    for c in candidates:
        by_cat.setdefault(c["category"], []).append(c)

    selected = []
    # Allocate proportionally: rate gets the most
    alloc = {"rate": 18, "percentage": 10, "proportion": 6, "arithmetic": 6}
    for cat, n in alloc.items():
        pool = by_cat.get(cat, [])
        random.shuffle(pool)
        selected.extend(pool[:n])

    # If we're short, fill from any remaining
    selected_ids = {p["source_id"] for p in selected}
    remaining = [c for c in candidates if c["source_id"] not in selected_ids]
    random.shuffle(remaining)
    while len(selected) < target and remaining:
        selected.append(remaining.pop(0))

    return selected[:target]


# ---------------------------------------------------------------------------
# MATH helpers
# ---------------------------------------------------------------------------


def _extract_math_answer(solution: str) -> float | None:
    r"""Extract numeric answer from \boxed{...} in MATH solution field."""
    m = re.search(r"\\boxed\{([^}]+)\}", solution)
    if not m:
        return None
    content = m.group(1).strip()
    # Only keep plain numbers (possibly negative, possibly decimal)
    content = content.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", content):
        try:
            return float(content)
        except ValueError:
            return None
    return None


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


def _categorize_math(problem: str, problem_type: str) -> str:
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
    # Default based on problem type field
    return "equation"


def pull_math(target: int = 30) -> list[dict]:
    """Pull and filter MATH problems (Algebra/Prealgebra, Level 1-3)."""
    ds = load_dataset("qwedsacf/competition_math", split="train")

    candidates = []
    for i, row in enumerate(ds):
        # Filter to Algebra and Prealgebra
        ptype = row.get("type", "")
        if ptype not in ("Algebra", "Prealgebra"):
            continue
        # Filter to Level 1-3
        level = row.get("level", "")
        level_num = None
        m = re.search(r"(\d+)", str(level))
        if m:
            level_num = int(m.group(1))
        if level_num is None or level_num > 3:
            continue

        answer = _extract_math_answer(row.get("solution", ""))
        if answer is None:
            continue

        question = row["problem"].strip()
        category = _categorize_math(question, ptype)
        candidates.append(
            {
                "question": question,
                "expected_answer": answer,
                "category": category,
                "source": "math",
                "source_id": f"math-train-{i}",
            }
        )

    # Sample with category diversity
    by_cat: dict[str, list[dict]] = {}
    for c in candidates:
        by_cat.setdefault(c["category"], []).append(c)

    selected = []
    alloc = {"equation": 12, "system": 6, "polynomial": 6, "inequality": 6}
    for cat, n in alloc.items():
        pool = by_cat.get(cat, [])
        random.shuffle(pool)
        selected.extend(pool[:n])

    # Fill remaining
    selected_ids = {p["source_id"] for p in selected}
    remaining = [c for c in candidates if c["source_id"] not in selected_ids]
    random.shuffle(remaining)
    while len(selected) < target and remaining:
        selected.append(remaining.pop(0))

    return selected[:target]


# ---------------------------------------------------------------------------
# Hand-curated problems
# ---------------------------------------------------------------------------

HAND_CURATED: list[dict] = [
    # === Compound Interest (4) ===
    {
        "question": "$10,000 invested at 5% annual interest compounded monthly. What is the value after 3 years?",
        "expected_answer": 11614.72,
        "category": "compound_interest",
        "source": "hand_curated",
        "source_id": "hand-compound-interest-1",
    },
    {
        "question": "$5,000 invested at 8% annual interest compounded quarterly. What is the value after 5 years?",
        "expected_answer": 7429.74,
        "category": "compound_interest",
        "source": "hand_curated",
        "source_id": "hand-compound-interest-2",
    },
    {
        "question": "$20,000 invested at 3.5% annual interest compounded monthly. What is the value after 10 years?",
        "expected_answer": 28366.90,
        "category": "compound_interest",
        "source": "hand_curated",
        "source_id": "hand-compound-interest-3",
    },
    {
        "question": "$1,000 invested at 12% annual interest compounded monthly. What is the value after 2 years?",
        "expected_answer": 1269.73,
        "category": "compound_interest",
        "source": "hand_curated",
        "source_id": "hand-compound-interest-4",
    },
    # === Kinematics (4) ===
    {
        "question": "A ball is thrown upward from 20 meters with velocity 15 m/s. What is the maximum height? (g=9.8 m/s^2)",
        "expected_answer": 31.48,
        "category": "kinematics",
        "source": "hand_curated",
        "source_id": "hand-kinematics-1",
    },
    {
        "question": "A car accelerates from rest at 3 m/s^2 for 8 seconds. How far does it travel?",
        "expected_answer": 96.0,
        "category": "kinematics",
        "source": "hand_curated",
        "source_id": "hand-kinematics-2",
    },
    {
        "question": "A projectile is launched at 25 m/s at 45 degrees. What is its range? (g=9.8 m/s^2)",
        "expected_answer": 63.78,
        "category": "kinematics",
        "source": "hand_curated",
        "source_id": "hand-kinematics-3",
    },
    {
        "question": "An object falls from 100 meters. How long until it hits the ground? (g=9.8 m/s^2)",
        "expected_answer": 4.52,
        "category": "kinematics",
        "source": "hand_curated",
        "source_id": "hand-kinematics-4",
    },
    # === Exponential Decay (4) ===
    {
        "question": "A substance has a half-life of 5 years. Starting with 200g, how much remains after 12 years?",
        "expected_answer": 37.15,
        "category": "exponential_decay",
        "source": "hand_curated",
        "source_id": "hand-exponential-1",
    },
    {
        "question": "A radioactive sample decays at 10% per year. Starting with 500g, how much remains after 8 years?",
        "expected_answer": 214.99,
        "category": "exponential_decay",
        "source": "hand_curated",
        "source_id": "hand-exponential-2",
    },
    {
        "question": "A population of bacteria doubles every 3 hours. Starting with 100, how many after 15 hours?",
        "expected_answer": 3200.0,
        "category": "exponential_decay",
        "source": "hand_curated",
        "source_id": "hand-exponential-3",
    },
    {
        "question": "A car depreciates by 15% per year. If it cost $30,000, what is it worth after 6 years?",
        "expected_answer": 11314.68,
        "category": "exponential_decay",
        "source": "hand_curated",
        "source_id": "hand-exponential-4",
    },
    # === Optimization (3) ===
    {
        "question": "A farmer has 200 meters of fencing. What is the maximum area of a rectangular pen he can enclose?",
        "expected_answer": 2500.0,
        "category": "optimization",
        "source": "hand_curated",
        "source_id": "hand-optimization-1",
    },
    {
        "question": "A box with a square base and open top must have a volume of 32 cm^3. What base side length minimizes the surface area?",
        "expected_answer": 4.0,
        "category": "optimization",
        "source": "hand_curated",
        "source_id": "hand-optimization-2",
    },
    {
        "question": "Find two positive numbers whose product is 64 and whose sum is minimized. What is their sum?",
        "expected_answer": 16.0,
        "category": "optimization",
        "source": "hand_curated",
        "source_id": "hand-optimization-3",
    },
]


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------


def _format_problem(p: dict) -> str:
    """Format a single problem dict as Python source."""
    # Use repr() so backslashes, quotes, newlines etc. are properly escaped
    q_repr = repr(p["question"])
    lines = [
        "    {",
        f'        "question": {q_repr},',
        f'        "expected_answer": {p["expected_answer"]},',
        f'        "category": "{p["category"]}",',
        f'        "source": "{p["source"]}",',
        f'        "source_id": "{p["source_id"]}",',
        "    },",
    ]
    return "\n".join(lines)


def generate_module(problems: list[dict]) -> str:
    """Generate the full dataset.py module source."""
    # Collect category stats
    by_source: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for p in problems:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1

    source_summary = ", ".join(f"{s}: {n}" for s, n in sorted(by_source.items()))
    cat_summary = ", ".join(f"{c}: {n}" for c, n in sorted(by_cat.items()))

    header = textwrap.dedent(f'''\
        """Calibration dataset -- math problems with known answers for prompt optimization.

        Built from established benchmarks with a hand-curated supplement for
        physics and finance problems not well-represented in standard datasets.

        Sources:
            - GSM8K (openai/gsm8k, train split) -- grade-school math word problems
            - MATH (qwedsacf/competition_math, train split) -- competition math, Algebra/Prealgebra Level 1-3
            - Hand-curated -- compound interest, kinematics, exponential decay, optimization

        Counts by source: {source_summary}
        Counts by category: {cat_summary}
        Total: {len(problems)} problems

        Generated by scripts/build_calibration_dataset.py (seed=42).
        """

        from __future__ import annotations

        _PROBLEMS: list[dict] = [
    ''')

    problem_blocks = "\n".join(_format_problem(p) for p in problems)

    footer = textwrap.dedent('''\
        ]


        def load_calibration_problems() -> list[dict]:
            """Return the calibration problem corpus.

            Each dict has keys: question, expected_answer, category, source, source_id.
            """
            return list(_PROBLEMS)
    ''')

    return header + problem_blocks + "\n" + footer


def main():
    print("Pulling GSM8K problems...")
    gsm8k_problems = pull_gsm8k(target=40)
    print(f"  Selected {len(gsm8k_problems)} GSM8K problems")

    print("Pulling MATH problems...")
    math_problems = pull_math(target=30)
    print(f"  Selected {len(math_problems)} MATH problems")

    print(f"Adding {len(HAND_CURATED)} hand-curated problems...")

    all_problems = gsm8k_problems + math_problems + HAND_CURATED
    print(f"Total: {len(all_problems)} problems")

    # Show category breakdown
    by_cat: dict[str, int] = {}
    for p in all_problems:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    print("\nCategory breakdown:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")

    by_source: dict[str, int] = {}
    for p in all_problems:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    print("\nSource breakdown:")
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")

    module_source = generate_module(all_problems)
    OUTPUT_PATH.write_text(module_source)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
