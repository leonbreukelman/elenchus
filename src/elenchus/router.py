"""Problem router — classifies math problems by domain, type, and complexity."""

from __future__ import annotations

import structlog

from elenchus import extract_json
from elenchus.config import get_model_config
from elenchus.llm import complete
from elenchus.state import RoutingResult

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a math problem classifier. Given a math problem, identify:

1. domain: the mathematical domain (algebra, calculus, statistics, arithmetic, geometry, etc.)
2. problem_type: specific type within the domain (linear_equation, quadratic, word_problem, etc.)
3. extracted_variables: list of variable names mentioned in the problem
4. complexity: low, medium, or high

Return ONLY valid JSON with these four fields. No markdown fences or extra text.
"""


async def route_problem(problem: str) -> RoutingResult:
    """Route a problem by classifying its domain and complexity.

    Uses the fast model tier — routing doesn't need the most capable model.
    """
    model = get_model_config().fast

    logger.debug("routing", problem=problem[:80])

    response = await complete(
        model=model,
        messages=[{"role": "user", "content": problem}],
        system=SYSTEM_PROMPT,
        max_tokens=get_model_config().max_tokens_fast,
    )

    data = extract_json(response.text)

    result = RoutingResult(
        domain=data["domain"],
        problem_type=data["problem_type"],
        extracted_variables=data["extracted_variables"],
        complexity=data["complexity"],
    )

    logger.info("routed", domain=result.domain, complexity=result.complexity)
    return result
