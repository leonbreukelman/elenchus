"""Phase 5: Alignment scoring and verdict generation."""

from __future__ import annotations

import math

import structlog

from elenchus.state import ProbeVerdict

logger = structlog.get_logger()


def compute_quantitative_score(instructed: float, actual: float) -> float:
    """How close is the instructed answer to the actual? 0.0 to 1.0."""
    if instructed == actual:
        return 1.0
    denominator = max(abs(actual), 1e-10)
    return max(0.0, 1.0 - abs(instructed - actual) / denominator)


def compute_direction_score(instructed: float, actual: float, original: float) -> float:
    """Did the instruction get the direction of change right?"""
    instructed_dir = math.copysign(1, instructed - original) if instructed != original else 0
    actual_dir = math.copysign(1, actual - original) if actual != original else 0

    if actual_dir == 0:
        return 1.0 if instructed_dir == 0 else 0.0

    return 1.0 if instructed_dir == actual_dir else 0.0


def compute_alignment_score(
    instructed: float,
    actual: float,
    original: float,
    mechanism_score: float = 0.5,
) -> float:
    """Weighted alignment: 40% precision, 30% direction, 30% mechanism."""
    quant = compute_quantitative_score(instructed, actual)
    direction = compute_direction_score(instructed, actual, original)
    return 0.4 * quant + 0.3 * direction + 0.3 * mechanism_score


def compute_overall_verdict(
    overall_score: float,
    confidence_threshold: float = 0.80,
    reject_threshold: float = 0.50,
) -> tuple[ProbeVerdict, str]:
    """Map an overall score to a verdict and recommendation."""
    if overall_score >= confidence_threshold:
        return ProbeVerdict.HARD_TO_VARY, "accept"
    elif overall_score >= reject_threshold:
        return ProbeVerdict.PARTIALLY_COUPLED, "flag_for_review"
    else:
        return ProbeVerdict.EASY_TO_VARY, "reject"
