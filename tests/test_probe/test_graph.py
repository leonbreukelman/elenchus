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


@pytest.mark.asyncio
async def test_ground_truth_proceeds_when_symbolic_matches_consensus(monkeypatch):
    """When symbolic answer matches consensus, ground truth should compute normally."""
    from unittest.mock import AsyncMock

    from elenchus.probe import graph as graph_module
    from elenchus.probe.graph import compute_ground_truths_node
    from elenchus.state import (
        ConsensusResult,
        Constraint,
        CouncilorResult,
        CouncilResult,
        Perturbation,
        RoutingResult,
    )
    from elenchus.tools.sandbox import SandboxResult

    symbolic_result = CouncilorResult(
        strategy="symbolic",
        answer=42.0,
        reasoning="correct",
        confidence=1.0,
        code="x = 10\nprint(42)",
    )
    consensus = ConsensusResult(
        answer=42.0,
        agreement="unanimous",
        confidence=1.0,
        dissenting_strategies=[],
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

    mock_gt = AsyncMock(return_value=SandboxResult(success=True, output="50.0", error=""))
    monkeypatch.setattr(graph_module, "compute_ground_truth", mock_gt)

    state = {
        "council_result": council_result,
        "perturbations": [perturbation],
    }

    result = await compute_ground_truths_node(state)
    assert len(result["ground_truths"]) == 1
    assert result["ground_truths"][0]["success"] is True
    mock_gt.assert_called_once()


@pytest.mark.asyncio
async def test_score_alignment_calls_mechanism_judge(monkeypatch):
    """The scoring node should call judge_mechanism for each prediction."""
    from unittest.mock import AsyncMock

    from elenchus.probe import mechanism_judge as mj_module
    from elenchus.probe.graph import score_alignment_node
    from elenchus.state import (
        ConsensusResult,
        Constraint,
        CouncilorResult,
        CouncilResult,
        Perturbation,
        RoutingResult,
    )

    mock_judge = AsyncMock(return_value=0.85)
    monkeypatch.setattr(mj_module, "judge_mechanism", mock_judge)

    constraint = Constraint(
        name="P",
        original_value=10000,
        dtype="numeric",
        role="principal",
        perturbation_range=(5000, 20000),
    )
    perturbation = Perturbation(
        constraint=constraint,
        new_value=15000,
        rationale="test",
    )
    consensus = ConsensusResult(
        answer=11614.72,
        agreement="unanimous",
        confidence=1.0,
        dissenting_strategies=[],
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
        councilor_results=[
            CouncilorResult(strategy="algebraic", answer=11614.72, reasoning="test", confidence=0.9),
        ],
        consensus=consensus,
    )

    state = {
        "council_result": council_result,
        "perturbations": [perturbation],
        "predictions": [
            {
                "_perturbation": "P",
                "_strategy": "algebraic",
                "new_answer": 17422.0,
                "new_reasoning": "Higher principal scales linearly.",
            }
        ],
        "ground_truths": [
            {
                "perturbation_name": "P",
                "new_value": 15000,
                "success": True,
                "output": "17422.08",
                "error": "",
            }
        ],
    }

    result = await score_alignment_node(state)
    mock_judge.assert_called_once()
    # reasoning_quality should be the judge's score, not 0.5
    sr = result["sensitivity_results"][0]
    assert sr.reasoning_quality == pytest.approx(0.85)
