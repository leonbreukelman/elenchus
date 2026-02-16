"""Tests for elenchus.state — all shared data models."""

from __future__ import annotations

import pytest

from elenchus.state import (
    ConsensusResult,
    Constraint,
    CouncilorResult,
    CouncilResult,
    DeutschProbeResult,
    Perturbation,
    ProbeVerdict,
    RoutingResult,
    SensitivityResult,
    VerifiedResult,
)

# ---------------------------------------------------------------------------
# ProbeVerdict enum
# ---------------------------------------------------------------------------


class TestProbeVerdict:
    def test_values(self):
        assert ProbeVerdict.HARD_TO_VARY == "hard_to_vary"
        assert ProbeVerdict.PARTIALLY_COUPLED == "partially_coupled"
        assert ProbeVerdict.EASY_TO_VARY == "easy_to_vary"

    def test_is_str(self):
        """ProbeVerdict members are also strings."""
        assert isinstance(ProbeVerdict.HARD_TO_VARY, str)

    def test_all_members(self):
        assert set(ProbeVerdict) == {
            ProbeVerdict.HARD_TO_VARY,
            ProbeVerdict.PARTIALLY_COUPLED,
            ProbeVerdict.EASY_TO_VARY,
        }


# ---------------------------------------------------------------------------
# RoutingResult
# ---------------------------------------------------------------------------


class TestRoutingResult:
    def test_construction(self):
        r = RoutingResult(
            domain="algebra",
            problem_type="equation",
            extracted_variables=["x", "y"],
            complexity="medium",
        )
        assert r.domain == "algebra"
        assert r.problem_type == "equation"
        assert r.extracted_variables == ["x", "y"]
        assert r.complexity == "medium"

    def test_empty_variables(self):
        r = RoutingResult(
            domain="calculus",
            problem_type="integral",
            extracted_variables=[],
            complexity="high",
        )
        assert r.extracted_variables == []

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            RoutingResult(domain="algebra", problem_type="equation")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# CouncilorResult
# ---------------------------------------------------------------------------


class TestCouncilorResult:
    def test_construction_with_defaults(self):
        c = CouncilorResult(
            strategy="symbolic",
            answer=42,
            reasoning="Solved via sympy.",
            confidence=0.95,
        )
        assert c.strategy == "symbolic"
        assert c.answer == 42
        assert c.reasoning == "Solved via sympy."
        assert c.code is None
        assert c.confidence == 0.95

    def test_code_optional(self):
        c = CouncilorResult(
            strategy="numeric",
            answer=3.14,
            reasoning="Approximated.",
            code="print(3.14)",
            confidence=0.8,
        )
        assert c.code == "print(3.14)"

    def test_answer_accepts_any_type(self):
        for val in [42, "x=2", [1, 2], {"a": 1}, None, True]:
            c = CouncilorResult(
                strategy="test",
                answer=val,
                reasoning="r",
                confidence=0.5,
            )
            assert c.answer == val


# ---------------------------------------------------------------------------
# ConsensusResult
# ---------------------------------------------------------------------------


class TestConsensusResult:
    def test_construction(self):
        c = ConsensusResult(
            answer=42,
            agreement="unanimous",
            confidence=0.99,
            dissenting_strategies=[],
        )
        assert c.answer == 42
        assert c.agreement == "unanimous"
        assert c.confidence == 0.99
        assert c.dissenting_strategies == []

    def test_with_dissent(self):
        c = ConsensusResult(
            answer=42,
            agreement="majority",
            confidence=0.75,
            dissenting_strategies=["monte_carlo"],
        )
        assert c.dissenting_strategies == ["monte_carlo"]


# ---------------------------------------------------------------------------
# CouncilResult
# ---------------------------------------------------------------------------


class TestCouncilResult:
    def _make_routing(self) -> RoutingResult:
        return RoutingResult(
            domain="algebra",
            problem_type="equation",
            extracted_variables=["x"],
            complexity="low",
        )

    def _make_councilor(self) -> CouncilorResult:
        return CouncilorResult(
            strategy="symbolic",
            answer=7,
            reasoning="Direct.",
            confidence=0.9,
        )

    def _make_consensus(self) -> ConsensusResult:
        return ConsensusResult(
            answer=7,
            agreement="unanimous",
            confidence=0.9,
            dissenting_strategies=[],
        )

    def test_construction(self):
        cr = CouncilResult(
            problem="Solve x+3=10",
            domain="algebra",
            routing=self._make_routing(),
            councilor_results=[self._make_councilor()],
            consensus=self._make_consensus(),
        )
        assert cr.problem == "Solve x+3=10"
        assert cr.domain == "algebra"
        assert len(cr.councilor_results) == 1

    def test_multiple_councilors(self):
        cr = CouncilResult(
            problem="Solve x+3=10",
            domain="algebra",
            routing=self._make_routing(),
            councilor_results=[self._make_councilor(), self._make_councilor()],
            consensus=self._make_consensus(),
        )
        assert len(cr.councilor_results) == 2


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


class TestConstraint:
    def test_construction(self):
        c = Constraint(
            name="initial_velocity",
            original_value=10.0,
            dtype="float",
            role="parameter",
            perturbation_range=(5.0, 15.0),
        )
        assert c.name == "initial_velocity"
        assert c.original_value == 10.0
        assert c.dtype == "float"
        assert c.role == "parameter"
        assert c.perturbation_range == (5.0, 15.0)

    def test_perturbation_range_tuple(self):
        c = Constraint(
            name="mass",
            original_value=1.0,
            dtype="float",
            role="variable",
            perturbation_range=(0.5, 2.0),
        )
        lo, hi = c.perturbation_range
        assert lo == 0.5
        assert hi == 2.0


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


class TestPerturbation:
    def _make_constraint(self) -> Constraint:
        return Constraint(
            name="mass",
            original_value=1.0,
            dtype="float",
            role="variable",
            perturbation_range=(0.5, 2.0),
        )

    def test_construction(self):
        p = Perturbation(
            constraint=self._make_constraint(),
            new_value=1.5,
            rationale="Testing sensitivity to mass.",
        )
        assert p.new_value == 1.5
        assert p.rationale == "Testing sensitivity to mass."
        assert p.constraint.name == "mass"


# ---------------------------------------------------------------------------
# SensitivityResult
# ---------------------------------------------------------------------------


class TestSensitivityResult:
    def _make_perturbation(self) -> Perturbation:
        c = Constraint(
            name="mass",
            original_value=1.0,
            dtype="float",
            role="variable",
            perturbation_range=(0.5, 2.0),
        )
        return Perturbation(constraint=c, new_value=1.5, rationale="test")

    def test_construction(self):
        sr = SensitivityResult(
            perturbation=self._make_perturbation(),
            predicted_answer=9.8,
            predicted_reasoning="F=ma with new mass.",
            actual_answer=9.81,
            alignment_score=0.99,
            reasoning_quality=0.85,
        )
        assert sr.predicted_answer == 9.8
        assert sr.actual_answer == 9.81
        assert sr.alignment_score == 0.99
        assert sr.reasoning_quality == 0.85


# ---------------------------------------------------------------------------
# DeutschProbeResult
# ---------------------------------------------------------------------------


class TestDeutschProbeResult:
    def _make_sensitivity_result(self) -> SensitivityResult:
        c = Constraint(
            name="mass",
            original_value=1.0,
            dtype="float",
            role="variable",
            perturbation_range=(0.5, 2.0),
        )
        p = Perturbation(constraint=c, new_value=1.5, rationale="test")
        return SensitivityResult(
            perturbation=p,
            predicted_answer=9.8,
            predicted_reasoning="F=ma",
            actual_answer=9.81,
            alignment_score=0.99,
            reasoning_quality=0.85,
        )

    def test_construction(self):
        dp = DeutschProbeResult(
            verdict=ProbeVerdict.HARD_TO_VARY,
            overall_score=0.95,
            sensitivity_map={"mass": 0.1, "velocity": 0.8},
            results=[self._make_sensitivity_result()],
            explanation_quality=0.9,
            recommendation="Explanation is well-coupled to constraints.",
            perturbation_log=[{"step": 1, "action": "perturb mass"}],
        )
        assert dp.verdict == ProbeVerdict.HARD_TO_VARY
        assert dp.overall_score == 0.95
        assert dp.sensitivity_map["mass"] == 0.1
        assert len(dp.results) == 1
        assert dp.explanation_quality == 0.9
        assert len(dp.perturbation_log) == 1

    def test_empty_results(self):
        dp = DeutschProbeResult(
            verdict=ProbeVerdict.EASY_TO_VARY,
            overall_score=0.2,
            sensitivity_map={},
            results=[],
            explanation_quality=0.1,
            recommendation="Explanation is not coupled.",
            perturbation_log=[],
        )
        assert dp.results == []
        assert dp.perturbation_log == []


# ---------------------------------------------------------------------------
# VerifiedResult
# ---------------------------------------------------------------------------


class TestVerifiedResult:
    def test_construction_minimal(self):
        vr = VerifiedResult(answer=42, confidence=0.9)
        assert vr.answer == 42
        assert vr.confidence == 0.9
        assert vr.explanation_quality is None
        assert vr.probe_verdict is None
        assert vr.sensitivity_map is None

    def test_construction_full(self):
        vr = VerifiedResult(
            answer=42,
            confidence=0.95,
            explanation_quality=0.88,
            probe_verdict=ProbeVerdict.HARD_TO_VARY,
            sensitivity_map={"mass": 0.1},
        )
        assert vr.explanation_quality == 0.88
        assert vr.probe_verdict == ProbeVerdict.HARD_TO_VARY
        assert vr.sensitivity_map == {"mass": 0.1}

    def test_optional_fields_default_to_none(self):
        vr = VerifiedResult(answer="x=5", confidence=0.7)
        assert vr.explanation_quality is None
        assert vr.probe_verdict is None
        assert vr.sensitivity_map is None
