"""Shared state models for the Elenchus verification engine.

All data structures that flow between pipeline stages are defined here.
Every other module imports from this single source of truth.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator


class ProbeVerdict(str, Enum):
    """Classification of explanation quality from the Deutsch probe."""

    HARD_TO_VARY = "hard_to_vary"
    PARTIALLY_COUPLED = "partially_coupled"
    EASY_TO_VARY = "easy_to_vary"


class RoutingResult(BaseModel):
    """Output of the domain router — classifies the problem."""

    domain: str
    problem_type: str
    extracted_variables: list[str]
    complexity: str


class CouncilorResult(BaseModel):
    """Output of a single councilor strategy."""

    strategy: str
    answer: Any
    reasoning: str
    code: str | None = None
    confidence: float


class ConsensusResult(BaseModel):
    """Aggregated consensus across councilors."""

    answer: Any
    agreement: str  # "unanimous", "majority", "arbitrated"
    confidence: float
    dissenting_strategies: list[str]


class CouncilResult(BaseModel):
    """Full output of the council phase."""

    problem: str
    domain: str
    routing: RoutingResult
    councilor_results: list[CouncilorResult]
    consensus: ConsensusResult


class ConstraintDtype(str, Enum):
    """Type classification for perturbation constraints."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    PROBABILITY = "probability"


class Constraint(BaseModel):
    """A named constraint extracted from the problem for perturbation testing."""

    name: str
    original_value: Any
    dtype: ConstraintDtype
    role: str
    perturbation_range: tuple[float, float]

    @field_validator("original_value", mode="before")
    @classmethod
    def _coerce_original_value(cls, v: object) -> object:
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            from elenchus import parse_number

            try:
                return parse_number(v)
            except ValueError:
                return v  # Let it through as string if parse fails
        return v

    @field_validator("perturbation_range", mode="before")
    @classmethod
    def _coerce_perturbation_range(cls, v: object) -> tuple[float, float] | object:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            coerced: list[float] = []
            for item in v:
                if isinstance(item, str):
                    from elenchus import parse_number

                    try:
                        coerced.append(parse_number(item))
                    except ValueError:
                        coerced.append(float(item))
                else:
                    coerced.append(item)
            return tuple(coerced)
        return v

    @field_validator("dtype", mode="before")
    @classmethod
    def _coerce_legacy_dtype(cls, v: str) -> str:
        if v in ("numeric", "float"):
            return "continuous"
        return v


class Perturbation(BaseModel):
    """A single perturbation applied to a constraint."""

    constraint: Constraint
    new_value: Any
    rationale: str


class SensitivityResult(BaseModel):
    """Result of testing one perturbation against the explanation."""

    perturbation: Perturbation
    instructed_answer: Any
    instructed_reasoning: str
    actual_answer: Any
    alignment_score: float
    reasoning_quality: float


class DeutschProbeResult(BaseModel):
    """Full output of the Deutsch probe phase."""

    verdict: ProbeVerdict
    overall_score: float
    sensitivity_map: dict[str, float]
    results: list[SensitivityResult]
    explanation_quality: float
    recommendation: str
    perturbation_log: list[dict]


class VerifiedResult(BaseModel):
    """Final pipeline output — the verified answer with quality metadata."""

    answer: Any
    confidence: float
    explanation_quality: float | None = None
    probe_verdict: ProbeVerdict | None = None
    sensitivity_map: dict[str, float] | None = None
