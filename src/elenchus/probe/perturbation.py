"""Phase 2: Generate perturbations for constraint testing."""

from __future__ import annotations

import random

import structlog

from elenchus.state import Constraint, Perturbation

logger = structlog.get_logger()


def _moderate_shift(constraint: Constraint) -> float:
    """30-50% shift from original value, staying within range."""
    lo, hi = constraint.perturbation_range
    original = float(constraint.original_value)
    shift_factor = random.uniform(0.3, 0.5) * random.choice([-1, 1])
    new_val = original * (1 + shift_factor)
    return max(lo, min(hi, new_val))


def _boundary_value(constraint: Constraint) -> float:
    """Push toward domain edge — either near min or max of range."""
    lo, hi = constraint.perturbation_range
    if random.random() < 0.5:
        return lo + (hi - lo) * random.uniform(0.0, 0.1)
    else:
        return hi - (hi - lo) * random.uniform(0.0, 0.1)


def _subtle_shift(constraint: Constraint) -> float:
    """5-10% shift for precision testing."""
    lo, hi = constraint.perturbation_range
    original = float(constraint.original_value)
    shift_factor = random.uniform(0.05, 0.10) * random.choice([-1, 1])
    new_val = original * (1 + shift_factor)
    return max(lo, min(hi, new_val))


def generate_perturbations(
    constraints: list[Constraint],
    budget: int = 3,
) -> list[Perturbation]:
    """Generate up to `budget` perturbations across constraints."""
    if not constraints:
        return []

    effective_budget = min(budget, len(constraints))
    perturbations: list[Perturbation] = []

    if effective_budget >= 1:
        c = constraints[0]
        perturbations.append(
            Perturbation(
                constraint=c,
                new_value=_moderate_shift(c),
                rationale="Primary sensitivity test — moderate shift on central constraint",
            )
        )

    if effective_budget >= 2:
        c = constraints[1 % len(constraints)]
        perturbations.append(
            Perturbation(
                constraint=c,
                new_value=_boundary_value(c),
                rationale="Boundary behavior test — errors amplify at extremes",
            )
        )

    if effective_budget >= 3:
        c = constraints[2 % len(constraints)]
        perturbations.append(
            Perturbation(
                constraint=c,
                new_value=_subtle_shift(c),
                rationale="Quantitative precision test — small shift tests magnitude accuracy",
            )
        )

    logger.info("perturbations_generated", count=len(perturbations))
    return perturbations
