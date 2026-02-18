"""Tests for elenchus.router — problem routing via LLM classifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from elenchus.llm import LLMResponse, UsageInfo
from elenchus.router import route_problem
from elenchus.state import RoutingResult


def _make_llm_response(data: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(data), model="test", usage=UsageInfo())


# ---------------------------------------------------------------------------
# route_problem
# ---------------------------------------------------------------------------


class TestRouteProblem:
    async def test_route_simple_equation(self):
        mock_response = _make_llm_response(
            {
                "domain": "algebra",
                "problem_type": "linear_equation",
                "extracted_variables": ["x"],
                "complexity": "low",
            }
        )

        with patch("elenchus.router.complete", new_callable=AsyncMock, return_value=mock_response) as mock_complete:
            result = await route_problem("Solve for x: 2x + 3 = 7")

        assert isinstance(result, RoutingResult)
        assert result.domain == "algebra"
        assert result.problem_type == "linear_equation"
        assert result.extracted_variables == ["x"]
        assert result.complexity == "low"

        mock_complete.assert_called_once()

    async def test_route_word_problem(self):
        mock_response = _make_llm_response(
            {
                "domain": "arithmetic",
                "problem_type": "word_problem",
                "extracted_variables": ["apples", "oranges"],
                "complexity": "medium",
            }
        )

        with patch("elenchus.router.complete", new_callable=AsyncMock, return_value=mock_response):
            result = await route_problem(
                "If John has 5 apples and gives away 2, then buys 3 oranges, how many pieces of fruit does he have?"
            )

        assert isinstance(result, RoutingResult)
        assert result.domain == "arithmetic"
        assert result.problem_type == "word_problem"
        assert result.extracted_variables == ["apples", "oranges"]
        assert result.complexity == "medium"
