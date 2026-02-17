"""Tests for elenchus.council.consensus — consensus engine with tolerance-aware matching."""

from __future__ import annotations

from elenchus.council.consensus import evaluate_consensus
from elenchus.state import CouncilorResult

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_result(strategy: str, answer: float | None, confidence: float) -> CouncilorResult:
    return CouncilorResult(
        strategy=strategy,
        answer=answer,
        reasoning=f"Solved via {strategy}",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# evaluate_consensus
# ---------------------------------------------------------------------------


class TestEvaluateConsensus:
    def test_unanimous_exact(self):
        """All councilors agree on the exact same answer."""
        results = [
            _make_result("algebraic", 42.0, 0.9),
            _make_result("numerical", 42.0, 0.85),
            _make_result("symbolic", 42.0, 1.0),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "unanimous"
        assert consensus.answer == 42.0
        assert consensus.dissenting_strategies == []
        assert consensus.confidence > 0

    def test_unanimous_within_tolerance(self):
        """Answers differ by less than rel_tol — still unanimous."""
        results = [
            _make_result("algebraic", 42.0, 0.9),
            _make_result("numerical", 42.0000001, 0.85),
            _make_result("symbolic", 42.0, 1.0),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "unanimous"
        assert consensus.dissenting_strategies == []

    def test_majority_two_of_three(self):
        """Two councilors agree, one dissents."""
        results = [
            _make_result("algebraic", 42.0, 0.9),
            _make_result("numerical", 42.0, 0.85),
            _make_result("symbolic", 99.0, 0.7),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "majority"
        assert consensus.answer == 42.0
        assert "symbolic" in consensus.dissenting_strategies
        assert len(consensus.dissenting_strategies) == 1

    def test_no_agreement(self):
        """All councilors give different answers."""
        results = [
            _make_result("algebraic", 10.0, 0.9),
            _make_result("numerical", 20.0, 0.85),
            _make_result("symbolic", 30.0, 0.7),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "none"
        # Picks highest-confidence answer
        assert consensus.answer == 10.0
        # Confidence is halved average
        avg = (0.9 + 0.85 + 0.7) / 3
        assert abs(consensus.confidence - avg / 2) < 1e-9

    def test_none_answer_excluded(self):
        """Results with answer=None are excluded from comparison."""
        results = [
            _make_result("algebraic", 42.0, 0.9),
            _make_result("numerical", 42.0, 0.85),
            _make_result("symbolic", None, 0.0),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "unanimous"
        assert consensus.answer == 42.0
        assert consensus.dissenting_strategies == []

    def test_confidence_average_in_unanimous(self):
        """Unanimous confidence is the average of all valid results."""
        results = [
            _make_result("algebraic", 7.0, 0.8),
            _make_result("numerical", 7.0, 0.6),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "unanimous"
        expected_confidence = (0.8 + 0.6) / 2
        assert abs(consensus.confidence - expected_confidence) < 1e-9

    def test_majority_confidence(self):
        """Majority confidence is the average of the agreeing group."""
        results = [
            _make_result("algebraic", 42.0, 0.8),
            _make_result("numerical", 42.0, 0.6),
            _make_result("symbolic", 99.0, 0.3),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "majority"
        expected_confidence = (0.8 + 0.6) / 2
        assert abs(consensus.confidence - expected_confidence) < 1e-9

    def test_custom_tolerance(self):
        """Custom rel_tol widens the matching window."""
        results = [
            _make_result("algebraic", 100.0, 0.9),
            _make_result("numerical", 101.0, 0.85),
            _make_result("symbolic", 100.0, 1.0),
        ]
        # Default tolerance: 101 is too far from 100
        consensus_tight = evaluate_consensus(results, rel_tol=1e-6)
        assert consensus_tight.agreement == "majority"

        # Wide tolerance: 101 matches 100
        consensus_wide = evaluate_consensus(results, rel_tol=0.02)
        assert consensus_wide.agreement == "unanimous"

    def test_all_none_answers(self):
        """When all answers are None, result is no agreement with None answer."""
        results = [
            _make_result("algebraic", None, 0.0),
            _make_result("numerical", None, 0.0),
        ]
        consensus = evaluate_consensus(results)
        assert consensus.agreement == "none"
        assert consensus.answer is None

    def test_consensus_with_wide_tolerance(self):
        """Answers within 0.1% should be unanimous at 1e-3 tolerance."""
        results = [
            _make_result("algebraic", 18.9368, 0.9),
            _make_result("numerical", 18.946, 0.85),
            _make_result("symbolic", 18.9400, 1.0),
        ]
        consensus = evaluate_consensus(results, rel_tol=1e-3)
        assert consensus.agreement == "unanimous"

    def test_consensus_rejects_genuine_disagreement_at_1e_3(self):
        """7.3% difference should still be caught at 1e-3 tolerance."""
        results = [
            _make_result("algebraic", 3.77, 0.8),
            _make_result("numerical", 4.065, 0.85),
            _make_result("symbolic", 4.065, 1.0),
        ]
        consensus = evaluate_consensus(results, rel_tol=1e-3)
        assert consensus.agreement == "majority"
        assert "algebraic" in consensus.dissenting_strategies
