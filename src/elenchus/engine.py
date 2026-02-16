"""Main Elenchus engine — LangGraph orchestrating router, council, and probe."""

from __future__ import annotations

import asyncio
import random
from typing import TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from elenchus.council.algebraic import AlgebraicCouncilor
from elenchus.council.consensus import evaluate_consensus
from elenchus.council.numerical import NumericalCouncilor
from elenchus.council.symbolic import SymbolicCouncilor
from elenchus.probe.graph import build_probe_graph
from elenchus.router import route_problem
from elenchus.state import (
    ConsensusResult,
    CouncilorResult,
    CouncilResult,
    DeutschProbeResult,
    RoutingResult,
    VerifiedResult,
)

logger = structlog.get_logger()


class EngineState(TypedDict, total=False):
    problem: str
    routing: RoutingResult
    councilor_results: list[CouncilorResult]
    consensus: ConsensusResult
    council_result: CouncilResult
    probe_result: DeutschProbeResult | None
    verified_result: VerifiedResult


async def route_node(state: EngineState) -> dict:
    routing = await route_problem(state["problem"])
    logger.info("routed", domain=routing.domain, complexity=routing.complexity)
    return {"routing": routing}


async def council_node(state: EngineState) -> dict:
    councilors = [AlgebraicCouncilor(), NumericalCouncilor(), SymbolicCouncilor()]
    results = await asyncio.gather(*[c.solve(state["problem"]) for c in councilors])
    return {"councilor_results": list(results)}


async def consensus_node(state: EngineState) -> dict:
    consensus = evaluate_consensus(state["councilor_results"])
    council_result = CouncilResult(
        problem=state["problem"],
        domain=state["routing"].domain,
        routing=state["routing"],
        councilor_results=state["councilor_results"],
        consensus=consensus,
    )
    return {"consensus": consensus, "council_result": council_result}


def should_probe(state: EngineState) -> str:
    """Decide whether to run the Deutsch Probe."""
    consensus = state["consensus"]
    routing = state["routing"]

    if routing.complexity in ("simple", "low") and consensus.agreement == "unanimous":
        return "skip_probe"

    if consensus.agreement == "unanimous":
        if random.random() < 0.30:
            return "run_probe"
        return "skip_probe"

    return "run_probe"


async def probe_node(state: EngineState) -> dict:
    probe_graph = build_probe_graph()
    result = await probe_graph.ainvoke({"council_result": state["council_result"]})
    return {"probe_result": result.get("probe_result")}


async def skip_probe_node(state: EngineState) -> dict:
    return {"probe_result": None}


async def output_node(state: EngineState) -> dict:
    probe = state.get("probe_result")
    verified = VerifiedResult(
        answer=state["consensus"].answer,
        confidence=state["consensus"].confidence,
        explanation_quality=probe.explanation_quality if probe else None,
        probe_verdict=probe.verdict if probe else None,
        sensitivity_map=probe.sensitivity_map if probe else None,
    )
    logger.info("verified_result", answer=verified.answer, probe_verdict=verified.probe_verdict)
    return {"verified_result": verified}


def build_engine_graph():
    """Build the main Elenchus engine graph."""
    builder = StateGraph(EngineState)

    builder.add_node("route", route_node)
    builder.add_node("council", council_node)
    builder.add_node("consensus", consensus_node)
    builder.add_node("probe", probe_node)
    builder.add_node("skip_probe", skip_probe_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "council")
    builder.add_edge("council", "consensus")
    builder.add_conditional_edges(
        "consensus",
        should_probe,
        {
            "run_probe": "probe",
            "skip_probe": "skip_probe",
        },
    )
    builder.add_edge("probe", "output")
    builder.add_edge("skip_probe", "output")
    builder.add_edge("output", END)

    return builder.compile()
