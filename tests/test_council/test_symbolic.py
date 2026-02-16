"""Tests for elenchus.council.symbolic — symbolic councilor with sandboxed SymPy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from elenchus.council.base import BaseCouncilor
from elenchus.council.symbolic import SymbolicCouncilor
from elenchus.state import CouncilorResult
from elenchus.tools.sandbox import SandboxResult

# ---------------------------------------------------------------------------
# SymbolicCouncilor
# ---------------------------------------------------------------------------


class TestSymbolicCouncilor:
    def test_is_base_councilor(self):
        assert issubclass(SymbolicCouncilor, BaseCouncilor)

    def test_strategy(self):
        c = SymbolicCouncilor()
        assert c.strategy == "symbolic"

    async def test_solve_success(self):
        """When sandbox executes SymPy code successfully, return parsed result."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "code": "from sympy import symbols, solve\nx = symbols('x')\nresult = solve(2*x + 3 - 17, x)\nprint(result[0])",
                        "reasoning": "Using SymPy to solve 2x + 3 = 17 algebraically",
                    }
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        sandbox_result = SandboxResult(success=True, output="7\n", error="")

        with (
            patch("elenchus.council.symbolic._get_client", return_value=mock_client),
            patch("elenchus.council.symbolic.execute_code", return_value=sandbox_result) as mock_exec,
        ):
            councilor = SymbolicCouncilor()
            result = await councilor.solve("Solve for x: 2x + 3 = 17")

        assert isinstance(result, CouncilorResult)
        assert result.strategy == "symbolic"
        assert result.answer == 7.0
        assert result.confidence == 1.0
        assert result.code is not None
        assert "sympy" in result.code
        mock_exec.assert_called_once()

    async def test_solve_sandbox_failure(self):
        """When sandbox fails, return result with answer=None and confidence=0."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "code": "from sympy import symbols, solve\nresult = bad_code()",
                        "reasoning": "Attempting to solve symbolically",
                    }
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        sandbox_result = SandboxResult(success=False, output="", error="NameError: name 'bad_code' is not defined")

        with (
            patch("elenchus.council.symbolic._get_client", return_value=mock_client),
            patch("elenchus.council.symbolic.execute_code", return_value=sandbox_result),
        ):
            councilor = SymbolicCouncilor()
            result = await councilor.solve("Solve for x: 2x + 3 = 17")

        assert isinstance(result, CouncilorResult)
        assert result.strategy == "symbolic"
        assert result.answer is None
        assert result.confidence == 0.0
        assert result.code is not None

    async def test_predict(self):
        """Predict uses LLM (not code execution) like other councilors."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "predicted_answer": 5.5,
                        "predicted_reasoning": "Changing constant from 3 to 6: 2x + 6 = 17 => x = 5.5",
                    }
                )
            )
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("elenchus.council.symbolic._get_client", return_value=mock_client):
            councilor = SymbolicCouncilor()
            result = await councilor.predict(
                problem="Solve for x: 2x + 3 = 17",
                original_answer=7.0,
                original_reasoning="SymPy: solve(2*x + 3 - 17, x) => 7",
                constraint_role="constant",
                original_value=3,
                new_value=6,
            )

        assert isinstance(result, dict)
        assert result["predicted_answer"] == 5.5
        assert "predicted_reasoning" in result
