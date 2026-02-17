"""Tests for the Deutsch Probe subgraph."""

import pytest

from elenchus.probe.graph import build_probe_graph


def test_probe_graph_builds():
    graph = build_probe_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_ground_truth_skipped_when_symbolic_answer_disagrees_with_consensus():
    """If symbolic answer doesn't match consensus, ground truth is unreliable — skip it."""
    from elenchus.probe.graph import compute_ground_truths_node
    from elenchus.state import (
        ConsensusResult,
        Constraint,
        CouncilorResult,
        CouncilResult,
        Perturbation,
        RoutingResult,
    )

    symbolic_result = CouncilorResult(
        strategy="symbolic",
        answer=999.0,  # Deliberately wrong — doesn't match consensus
        reasoning="wrong code",
        confidence=1.0,
        code="x = 10\nprint(x)",
    )
    consensus = ConsensusResult(
        answer=42.0,  # Consensus says 42
        agreement="majority",
        confidence=0.7,
        dissenting_strategies=["symbolic"],
    )
    council_result = CouncilResult(
        problem="test",
        domain="algebra",
        routing=RoutingResult(
            domain="algebra",
            problem_type="equation",
            extracted_variables=[],
            complexity="medium",
        ),
        councilor_results=[symbolic_result],
        consensus=consensus,
    )
    constraint = Constraint(
        name="x",
        original_value=10,
        dtype="numeric",
        role="test var",
        perturbation_range=(5, 20),
    )
    perturbation = Perturbation(
        constraint=constraint,
        new_value=15,
        rationale="test",
    )

    state = {
        "council_result": council_result,
        "perturbations": [perturbation],
    }

    result = await compute_ground_truths_node(state)
    assert result["ground_truths"] == []
