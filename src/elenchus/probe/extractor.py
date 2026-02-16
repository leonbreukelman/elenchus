"""Phase 1: Extract perturbable constraints from a problem."""

from __future__ import annotations

import json

import anthropic
import structlog

from elenchus.state import Constraint

logger = structlog.get_logger()

EXTRACTION_PROMPT = """\
Given this math problem and its solution, identify every input constraint that the answer depends on.

For each, return a JSON array of objects with:
- name: variable name (snake_case)
- original_value: the numeric value
- dtype: "numeric" (for now, always numeric)
- role: plain English description of what this constraint represents
- perturbation_range: [min, max] of valid alternative values

Only include constraints where changing the value SHOULD change the answer.
Exclude constants of nature and fixed definitions.

Return ONLY a valid JSON array.

Problem: {problem}
Solution: {solution}"""


def _get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


async def extract_constraints(problem: str, solution: str) -> list[Constraint]:
    """Extract perturbable constraints from a problem-solution pair."""
    client = _get_client()
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(problem=problem, solution=solution),
            }
        ],
    )
    raw = response.content[0].text
    logger.info("constraint_extraction", raw=raw)
    parsed = json.loads(raw)
    return [Constraint(**c) for c in parsed]
