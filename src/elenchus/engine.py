"""Main Elenchus engine — LangGraph orchestrating router, council, and probe."""

from __future__ import annotations

import asyncio
from typing import TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from elenchus.council.algebraic import AlgebraicCouncilor
from elenchus.council.consensus import evaluate_consensus
from elenchus.council.numerical import NumericalCouncilor
from elenchus.council.symbolic import SymbolicCouncilor
from elenchus.llm import UsageInfo
from elenchus.policy.loader import load_domain_config
from elenchus.policy.schemas import DomainConfig
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


class UsageStats(BaseModel):
    """Accumulated token usage and cost across all LLM calls."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0

    def add(self, usage: UsageInfo) -> None:
        """Accumulate usage from a single LLM call."""
        self.total_tokens += usage.total_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_cost_usd += usage.cost_usd
        self.calls += 1


class EngineState(TypedDict, total=False):
    problem: str
    routing: RoutingResult
    domain_config: DomainConfig
    councilor_results: list[CouncilorResult]
    consensus: ConsensusResult
    council_result: CouncilResult
    probe_result: DeutschProbeResult | None
    verified_result: VerifiedResult
    usage: UsageStats


async def route_node(state: EngineState) -> dict:
    routing = await route_problem(state["problem"])
    logger.info("routed", domain=routing.domain, complexity=routing.complexity)
    return {"routing": routing}


async def council_node(state: EngineState) -> dict:
    councilors = [AlgebraicCouncilor(), NumericalCouncilor(), SymbolicCouncilor()]
    raw_results = await asyncio.gather(
        *[c.solve(state["problem"]) for c in councilors],
        return_exceptions=True,
    )
    results = []
    for r, c in zip(raw_results, councilors):
        if isinstance(r, Exception):
            logger.warning("councilor_failed", strategy=c.strategy, error=str(r))
        else:
            results.append(r)
    if not results:
        raise RuntimeError("All councilors failed — cannot proceed")
    return {"councilor_results": results}


async def consensus_node(state: EngineState) -> dict:
    domain_name = state["routing"].domain
    try:
        domain_config = load_domain_config(domain_name)
    except FileNotFoundError:
        logger.info("domain_config_fallback", domain=domain_name)
        domain_config = load_domain_config("_base")

    rel_tol = domain_config.council.consensus_tolerance_relative
    consensus = evaluate_consensus(state["councilor_results"], rel_tol=rel_tol)
    logger.info("consensus_evaluated", agreement=consensus.agreement, tolerance=rel_tol)

    council_result = CouncilResult(
        problem=state["problem"],
        domain=domain_name,
        routing=state["routing"],
        councilor_results=state["councilor_results"],
        consensus=consensus,
    )
    return {"consensus": consensus, "council_result": council_result, "domain_config": domain_config}


async def probe_node(state: EngineState) -> dict:
    domain_config = state.get("domain_config")
    probe_config = domain_config.probe if domain_config else None

    probe_graph = build_probe_graph()
    probe_input = {"council_result": state["council_result"]}
    if probe_config:
        probe_input["perturbation_budget"] = probe_config.perturbation_budget
        probe_input["confidence_threshold"] = probe_config.confidence_threshold
        probe_input["reject_threshold"] = probe_config.reject_threshold

    result = await probe_graph.ainvoke(probe_input)
    return {"probe_result": result.get("probe_result")}


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
    builder.add_node("output", output_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "council")
    builder.add_edge("council", "consensus")
    builder.add_edge("consensus", "probe")
    builder.add_edge("probe", "output")
    builder.add_edge("output", END)

    return builder.compile()
