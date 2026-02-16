"""Symbolic councilor — generates SymPy code and executes it in a sandbox."""

from __future__ import annotations

import json
from typing import Any

import anthropic
import structlog

from elenchus.council.base import BaseCouncilor
from elenchus.state import CouncilorResult
from elenchus.tools.sandbox import execute_code

log = structlog.get_logger()

SOLVE_MODEL = "claude-sonnet-4-5-20250929"

SOLVE_PROMPT = """\
You are a mathematical solver that generates SymPy Python code.
Given a math problem, write Python code using SymPy to solve it.
The code MUST print the final numeric answer as the last line of output.

Return ONLY valid JSON with:
- "code": the complete Python code (using sympy, must print the answer)
- "reasoning": brief explanation of the approach

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


def _parse_numeric_output(output: str) -> float:
    """Extract the last line of sandbox output and convert to float."""
    lines = output.strip().splitlines()
    last_line = lines[-1].strip()
    return float(last_line)


class SymbolicCouncilor(BaseCouncilor):
    """Councilor that generates SymPy code and executes it in a sandbox."""

    strategy: str = "symbolic"

    async def solve(self, problem: str) -> CouncilorResult:
        """Generate SymPy code, execute in sandbox, return parsed result."""
        client = _get_client()

        log.debug("symbolic.solve", problem=problem[:80])

        response = await client.messages.create(
            model=SOLVE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": problem}],
            system=SOLVE_PROMPT,
        )

        raw = response.content[0].text
        log.debug("symbolic.raw_response", raw=raw[:200])

        data = json.loads(raw)
        code = data["code"]
        reasoning = data["reasoning"]

        log.debug("symbolic.executing_code", code_len=len(code))
        sandbox_result = await execute_code(code)

        if sandbox_result.success:
            try:
                answer = _parse_numeric_output(sandbox_result.output)
            except (ValueError, IndexError):
                log.warning("symbolic.parse_failed", output=sandbox_result.output[:200])
                return CouncilorResult(
                    strategy=self.strategy,
                    answer=None,
                    reasoning=reasoning,
                    confidence=0.0,
                    code=code,
                )

            log.info("symbolic.solved", answer=answer)
            return CouncilorResult(
                strategy=self.strategy,
                answer=answer,
                reasoning=reasoning,
                confidence=1.0,
                code=code,
            )

        log.warning("symbolic.sandbox_failed", error=sandbox_result.error[:200])
        return CouncilorResult(
            strategy=self.strategy,
            answer=None,
            reasoning=reasoning,
            confidence=0.0,
            code=code,
        )

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

        log.debug("symbolic.predict", constraint_role=constraint_role, new_value=new_value)

        response = await client.messages.create(
            model=SOLVE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            system="You are a precise mathematical predictor. Return only valid JSON.",
        )

        raw = response.content[0].text
        log.debug("symbolic.predict_response", raw=raw[:200])

        return json.loads(raw)
