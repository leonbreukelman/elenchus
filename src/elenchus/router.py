"""Problem router — classifies math problems using a Haiku-class LLM call."""

from __future__ import annotations

import json

import anthropic
import structlog

from elenchus.state import RoutingResult

log = structlog.get_logger()

ROUTING_MODEL = "claude-haiku-4-5-20251001"

ROUTING_PROMPT = """\
You are a math problem classifier. Analyze the given problem and return a JSON object with:

- "domain": the mathematical domain (e.g. "algebra", "calculus", "arithmetic", "geometry", "statistics")
- "problem_type": specific type within the domain (e.g. "linear_equation", "quadratic", "word_problem", "integral")
- "extracted_variables": list of variable names or key quantities mentioned in the problem
- "complexity": one of "low", "medium", "high"

Return ONLY valid JSON, no markdown fences or extra text.
"""


def _get_client() -> anthropic.AsyncAnthropic:
    """Factory for the Anthropic async client. Isolated for easy mocking."""
    return anthropic.AsyncAnthropic()


async def route_problem(problem: str) -> RoutingResult:
    """Classify a math problem by domain, type, variables, and complexity.

    Sends the problem to a Haiku-class model and parses the JSON response
    into a :class:`RoutingResult`.
    """
    client = _get_client()

    log.debug("router.classifying", problem=problem[:80])

    response = await client.messages.create(
        model=ROUTING_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": problem}],
        system=ROUTING_PROMPT,
    )

    raw = response.content[0].text
    log.debug("router.raw_response", raw=raw[:200])

    data = json.loads(raw)

    result = RoutingResult(
        domain=data["domain"],
        problem_type=data["problem_type"],
        extracted_variables=data["extracted_variables"],
        complexity=data["complexity"],
    )

    log.info(
        "router.classified",
        domain=result.domain,
        problem_type=result.problem_type,
        complexity=result.complexity,
    )

    return result
