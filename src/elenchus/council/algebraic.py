"""Algebraic councilor — solves problems via step-by-step algebraic reasoning."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import structlog

from elenchus.council.base import BaseCouncilor
from elenchus.state import CouncilorResult

log = structlog.get_logger()

SOLVE_MODEL = "claude-sonnet-4-5-20250929"

SOLVE_PROMPT = """\
You are a precise mathematical solver using algebraic methods.
Solve the given problem step by step using algebraic manipulation.

Return ONLY valid JSON with:
- "answer": the numeric answer (a number, not a string)
- "reasoning": step-by-step algebraic solution
- "confidence": your confidence from 0.0 to 1.0

No markdown fences or extra text.
"""

PREDICT_PROMPT = """\
You previously solved this problem:

Problem: {problem}
Your answer: {original_answer}
Your reasoning: {original_reasoning}

Now a constraint has changed. The {constraint_role} "{original_value}" \
has been changed to "{new_value}".

Predict what the new answer would be under this change.
Return ONLY valid JSON with:
- "predicted_answer": the new numeric answer
- "predicted_reasoning": step-by-step reasoning for the new answer

No markdown fences or extra text.
"""


def _get_client() -> anthropic.AsyncAnthropic:
    """Factory for the Anthropic async client. Isolated for easy mocking."""
    return anthropic.AsyncAnthropic()


class AlgebraicCouncilor(BaseCouncilor):
    """Councilor that solves problems via algebraic manipulation."""

    strategy: str = "algebraic"

    async def solve(self, problem: str) -> CouncilorResult:
        """Solve the problem using step-by-step algebra."""
        client = _get_client()

        log.debug("algebraic.solve", problem=problem[:80])

        response = await client.messages.create(
            model=SOLVE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": problem}],
            system=SOLVE_PROMPT,
        )

        raw = response.content[0].text
        log.debug("algebraic.raw_response", raw=raw[:200])

        data = json.loads(raw)

        result = CouncilorResult(
            strategy=self.strategy,
            answer=data["answer"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            code=None,
        )

        log.info("algebraic.solved", answer=result.answer, confidence=result.confidence)
        return result

    async def predict(
        self,
        problem: str,
        original_answer: Any,
        original_reasoning: str,
        constraint_role: str,
        original_value: Any,
        new_value: Any,
    ) -> dict:
        """Predict how the answer changes under a constraint perturbation."""
        client = _get_client()

        prompt = PREDICT_PROMPT.format(
            problem=problem,
            original_answer=original_answer,
            original_reasoning=original_reasoning,
            constraint_role=constraint_role,
            original_value=original_value,
            new_value=new_value,
        )

        log.debug("algebraic.predict", constraint_role=constraint_role, new_value=new_value)

        response = await client.messages.create(
            model=SOLVE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            system="You are a precise mathematical predictor. Return only valid JSON.",
        )

        raw = response.content[0].text
        log.debug("algebraic.predict_response", raw=raw[:200])

        return json.loads(raw)
