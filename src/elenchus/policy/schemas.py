"""Pydantic models for domain policy configuration.

Each domain (e.g. algebra, calculus) has a DomainConfig that controls
probe behaviour, council strategies, and router settings. Configs are
loaded from YAML files and support inheritance via the ``extends`` key.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SandboxConfig(BaseModel):
    """Controls for the code-execution sandbox."""

    timeout: int = 30
    allowed_imports: list[str] = Field(default_factory=lambda: ["sympy", "numpy"])


class ProbeConfig(BaseModel):
    """Tuning knobs for the Deutsch probe phase."""

    sample_rate_unanimous: float = 0.30
    perturbation_budget: int = 3
    confidence_threshold: float = 0.80
    reject_threshold: float = 0.50
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    preferred_perturbations: list[dict] = Field(default_factory=list)


class CouncilConfig(BaseModel):
    """Council phase configuration — strategies and consensus rules."""

    strategies: list[str] = Field(default_factory=lambda: ["algebraic", "numerical", "symbolic"])
    consensus_tolerance_relative: float = 1e-3
    debate_rounds: int = 1


class RouterConfig(BaseModel):
    """Domain router configuration."""

    problem_types: list[str] = Field(default_factory=list)


class DomainConfig(BaseModel):
    """Top-level domain configuration aggregating all sub-configs."""

    name: str = "_base"
    extends: str | None = None
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
