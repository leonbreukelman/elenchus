import pytest

from elenchus.council.base import BaseCouncilor
from elenchus.probe.instructor import collect_instructions
from elenchus.state import Constraint, CouncilorResult, Perturbation


class MockCouncilor(BaseCouncilor):
    strategy = "mock"

    async def solve(self, problem):
        pass

    async def instruct(self, problem, original_answer, original_reasoning, constraint_role, original_value, new_value):
        return {
            "direction": "increase",
            "new_answer": original_answer * 1.5,
            "mechanism": "mock mechanism",
            "confidence": "high",
        }


@pytest.mark.asyncio
async def test_collect_instructions_basic():
    councilors = [MockCouncilor(), MockCouncilor()]
    results = [
        CouncilorResult(strategy="mock", answer=100.0, reasoning="test", confidence=0.9),
        CouncilorResult(strategy="mock", answer=100.0, reasoning="test", confidence=0.9),
    ]
    constraint = Constraint(
        name="rate",
        original_value=0.05,
        dtype="numeric",
        role="annual rate",
        perturbation_range=(0.01, 0.20),
    )
    perturbation = Perturbation(constraint=constraint, new_value=0.08, rationale="test")

    instructions = await collect_instructions(
        councilors=councilors,
        councilor_results=results,
        perturbations=[perturbation],
        problem="test problem",
    )

    assert len(instructions) == 2
    assert all(p["direction"] == "increase" for p in instructions)


@pytest.mark.asyncio
async def test_instructions_parallel():
    """Verify multiple instructions run in parallel."""
    councilors = [MockCouncilor(), MockCouncilor(), MockCouncilor()]
    results = [
        CouncilorResult(strategy="mock", answer=100.0, reasoning="test", confidence=0.9),
        CouncilorResult(strategy="mock", answer=100.0, reasoning="test", confidence=0.9),
        CouncilorResult(strategy="mock", answer=100.0, reasoning="test", confidence=0.9),
    ]
    constraint = Constraint(
        name="rate",
        original_value=0.05,
        dtype="numeric",
        role="rate",
        perturbation_range=(0.01, 0.20),
    )
    perturbations = [
        Perturbation(constraint=constraint, new_value=0.08, rationale="test1"),
        Perturbation(constraint=constraint, new_value=0.02, rationale="test2"),
    ]

    instructions = await collect_instructions(
        councilors=councilors,
        councilor_results=results,
        perturbations=perturbations,
        problem="test",
    )

    assert len(instructions) == 6
