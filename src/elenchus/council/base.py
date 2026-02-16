"""Abstract base class for councilor strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from elenchus.state import CouncilorResult


class BaseCouncilor(ABC):
    """Base class that all councilor strategies extend.

    Each councilor implements a distinct problem-solving strategy and
    can also predict how its answer would change under perturbation.
    """

    strategy: str

    @abstractmethod
    async def solve(self, problem: str) -> CouncilorResult:
        """Solve the given math problem and return a structured result."""

    @abstractmethod
    async def predict(
        self,
        problem: str,
        original_answer: Any,
        original_reasoning: str,
        constraint_role: str,
        original_value: Any,
        new_value: Any,
    ) -> dict:
        """Predict how the answer changes when a constraint is perturbed."""
