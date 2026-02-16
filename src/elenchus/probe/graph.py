"""Deutsch Probe subgraph — independent LangGraph for perturbation verification."""

from __future__ import annotations

from typing import TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from elenchus.council.algebraic import AlgebraicCouncilor
from elenchus.council.numerical import NumericalCouncilor
from elenchus.council.symbolic import SymbolicCouncilor
from elenchus.probe.extractor import extract_constraints
from elenchus.probe.ground_truth import compute_ground_truth
from elenchus.probe.perturbation import generate_perturbations
from elenchus.probe.predictor import collect_predictions
from elenchus.probe.scorer import compute_alignment_score, compute_overall_verdict
from elenchus.state import (
    Constraint,
    CouncilResult,
    DeutschProbeResult,
    Perturbation,
    SensitivityResult,
)

logger = structlog.get_logger()

_COUNCILOR_MAP = {
    "algebraic": AlgebraicCouncilor,
    "numerical": NumericalCouncilor,
    "symbolic": SymbolicCouncilor,
}


class ProbeState(TypedDict, total=False):
    council_result: CouncilResult
    constraints: list[Constraint]
    perturbations: list[Perturbation]
    predictions: list[dict]
    ground_truths: list[dict]
    sensitivity_results: list[SensitivityResult]
    probe_result: DeutschProbeResult


async def extract_constraints_node(state: ProbeState) -> dict:
    cr = state["council_result"]
    symbolic_result = next(
        (r for r in cr.councilor_results if r.strategy == "symbolic" and r.code),
        None,
    )
    constraints = await extract_constraints(
        problem=cr.problem,
        solution=str(cr.consensus.answer),
        symbolic_code=symbolic_result.code if symbolic_result else None,
    )
    return {"constraints": constraints}


async def generate_perturbations_node(state: ProbeState) -> dict:
    perturbations = generate_perturbations(state["constraints"], budget=3)
    return {"perturbations": perturbations}


async def collect_predictions_node(state: ProbeState) -> dict:
    cr = state["council_result"]
    councilors = []
    for result in cr.councilor_results:
        cls = _COUNCILOR_MAP.get(result.strategy)
        if cls:
            councilors.append(cls())

    predictions = await collect_predictions(
        councilors=councilors,
        councilor_results=cr.councilor_results,
        perturbations=state["perturbations"],
        problem=cr.problem,
    )
    return {"predictions": predictions}


async def compute_ground_truths_node(state: ProbeState) -> dict:
    cr = state["council_result"]
    symbolic_result = next(
        (r for r in cr.councilor_results if r.strategy == "symbolic" and r.code),
        None,
    )
    if not symbolic_result:
        logger.warning("no_symbolic_code_for_ground_truth")
        return {"ground_truths": []}

    ground_truths = []
    for perturbation in state["perturbations"]:
        result = await compute_ground_truth(symbolic_result.code, perturbation)
        ground_truths.append(
            {
                "perturbation_name": perturbation.constraint.name,
                "new_value": perturbation.new_value,
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }
        )
    return {"ground_truths": ground_truths}


async def score_alignment_node(state: ProbeState) -> dict:
    cr = state["council_result"]
    original_answer = float(cr.consensus.answer)

    sensitivity_results = []
    sensitivity_map: dict[str, float] = {}

    for gt in state["ground_truths"]:
        if not gt["success"]:
            continue

        actual = float(gt["output"])
        constraint_name = gt["perturbation_name"]

        perturbation = next(
            (p for p in state["perturbations"] if p.constraint.name == constraint_name),
            None,
        )
        if not perturbation:
            continue

        matching_preds = [p for p in state["predictions"] if p.get("_perturbation") == constraint_name]

        for pred in matching_preds:
            try:
                predicted = float(pred.get("predicted_answer", 0))
            except (ValueError, TypeError):
                predicted = 0.0

            score = compute_alignment_score(
                predicted=predicted,
                actual=actual,
                original=original_answer,
            )

            sensitivity_results.append(
                SensitivityResult(
                    perturbation=perturbation,
                    predicted_answer=predicted,
                    predicted_reasoning=pred.get("mechanism", ""),
                    actual_answer=actual,
                    alignment_score=score,
                    reasoning_quality=0.5,
                )
            )

        if matching_preds:
            scores = [
                sr.alignment_score for sr in sensitivity_results if sr.perturbation.constraint.name == constraint_name
            ]
            sensitivity_map[constraint_name] = sum(scores) / len(scores) if scores else 0.0

    overall = sum(sensitivity_map.values()) / len(sensitivity_map) if sensitivity_map else 0.0
    verdict, recommendation = compute_overall_verdict(overall)

    probe_result = DeutschProbeResult(
        verdict=verdict,
        overall_score=overall,
        sensitivity_map=sensitivity_map,
        results=sensitivity_results,
        explanation_quality=overall,
        recommendation=recommendation,
        perturbation_log=[gt for gt in state["ground_truths"]],
    )

    logger.info("probe_verdict", verdict=verdict.value, score=overall, recommendation=recommendation)
    return {"probe_result": probe_result, "sensitivity_results": sensitivity_results}


def build_probe_graph():
    """Build the Deutsch Probe subgraph."""
    builder = StateGraph(ProbeState)

    builder.add_node("extract_constraints", extract_constraints_node)
    builder.add_node("generate_perturbations", generate_perturbations_node)
    builder.add_node("collect_predictions", collect_predictions_node)
    builder.add_node("compute_ground_truths", compute_ground_truths_node)
    builder.add_node("score_alignment", score_alignment_node)

    builder.add_edge(START, "extract_constraints")
    builder.add_edge("extract_constraints", "generate_perturbations")
    builder.add_edge("generate_perturbations", "collect_predictions")
    builder.add_edge("generate_perturbations", "compute_ground_truths")
    builder.add_edge("collect_predictions", "score_alignment")
    builder.add_edge("compute_ground_truths", "score_alignment")
    builder.add_edge("score_alignment", END)

    return builder.compile()
