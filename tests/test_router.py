"""Tests for elenchus.router — problem routing via Haiku classifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from elenchus.router import route_problem
from elenchus.state import RoutingResult

# ---------------------------------------------------------------------------
# route_problem
# ---------------------------------------------------------------------------


class TestRouteProblem:
    async def test_route_simple_equation(self):
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "domain": "algebra",
                        "problem_type": "linear_equation",
                        "extracted_variables": ["x"],
                        "complexity": "low",
                    }
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("elenchus.router._get_client", return_value=mock_client):
            result = await route_problem("Solve for x: 2x + 3 = 7")

        assert isinstance(result, RoutingResult)
        assert result.domain == "algebra"
        assert result.problem_type == "linear_equation"
        assert result.extracted_variables == ["x"]
        assert result.complexity == "low"

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    async def test_route_word_problem(self):
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "domain": "arithmetic",
                        "problem_type": "word_problem",
                        "extracted_variables": ["apples", "oranges"],
                        "complexity": "medium",
                    }
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("elenchus.router._get_client", return_value=mock_client):
            result = await route_problem(
                "If John has 5 apples and gives away 2, then buys 3 oranges, how many pieces of fruit does he have?"
            )

        assert isinstance(result, RoutingResult)
        assert result.domain == "arithmetic"
        assert result.problem_type == "word_problem"
        assert result.extracted_variables == ["apples", "oranges"]
        assert result.complexity == "medium"
