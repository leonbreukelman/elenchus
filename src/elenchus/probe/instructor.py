"""Phase 3: Collect councilor responses for perturbations."""

from __future__ import annotations

import asyncio

import structlog

from elenchus.council.base import BaseCouncilor
from elenchus.state import CouncilorResult, Perturbation

logger = structlog.get_logger()


async def _single_instruction(
    councilor: BaseCouncilor,
    councilor_result: CouncilorResult,
    perturbation: Perturbation,
    problem: str,
) -> dict:
    """Get one councilor's response for one perturbation."""
    result = await councilor.instruct(
        problem=problem,
        original_answer=councilor_result.answer,
        original_reasoning=councilor_result.reasoning,
        constraint_role=perturbation.constraint.role,
        original_value=perturbation.constraint.original_value,
        new_value=perturbation.new_value,
    )
    result["_strategy"] = councilor.strategy
    result["_perturbation"] = perturbation.constraint.name
    return result


async def collect_instructions(
    councilors: list[BaseCouncilor],
    councilor_results: list[CouncilorResult],
    perturbations: list[Perturbation],
    problem: str,
) -> list[dict]:
    """Collect all councilor instructions for all perturbations in parallel."""
    tasks = []
    for councilor, result in zip(councilors, councilor_results):
        for perturbation in perturbations:
            tasks.append(_single_instruction(councilor, result, perturbation, problem))

    results = await asyncio.gather(*tasks)
    logger.info("instructions_collected", count=len(results))
    return list(results)
