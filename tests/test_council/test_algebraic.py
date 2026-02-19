"""Tests for elenchus.council.algebraic — algebraic councilor strategy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elenchus.council.algebraic import AlgebraicCouncilor
from elenchus.council.base import BaseCouncilor
from elenchus.llm import LLMResponse, UsageInfo
from elenchus.state import CouncilorResult


def _make_llm_response(data: dict) -> LLMResponse:
    """Helper to build an LLMResponse from a dict."""
    return LLMResponse(text=json.dumps(data), model="test", usage=UsageInfo())


# ---------------------------------------------------------------------------
# AlgebraicCouncilor
# ---------------------------------------------------------------------------


class TestAlgebraicCouncilor:
    def test_is_base_councilor(self):
        assert issubclass(AlgebraicCouncilor, BaseCouncilor)

    def test_strategy(self):
        c = AlgebraicCouncilor()
        assert c.strategy == "algebraic"

    async def test_solve(self):
        mock_response = _make_llm_response(
            {
                "answer": 7.0,
                "reasoning": "2x + 3 = 17 => 2x = 14 => x = 7",
                "confidence": 0.95,
            }
        )

        with patch("elenchus.council.base.complete", new_callable=AsyncMock, return_value=mock_response):
            councilor = AlgebraicCouncilor()
            result = await councilor.solve("Solve for x: 2x + 3 = 17")

        assert isinstance(result, CouncilorResult)
        assert result.strategy == "algebraic"
        assert result.answer == 7.0
        assert result.reasoning == "2x + 3 = 17 => 2x = 14 => x = 7"
        assert result.confidence == 0.95
        assert result.code is None

    async def test_instruct(self):
        mock_response = _make_llm_response(
            {
                "new_answer": 10.0,
                "new_reasoning": "If the constant changes from 3 to 6, 2x + 6 = 17 => x = 5.5",
            }
        )

        with patch("elenchus.council.base.complete", new_callable=AsyncMock, return_value=mock_response):
            councilor = AlgebraicCouncilor()
            result = await councilor.instruct(
                problem="Solve for x: 2x + 3 = 17",
                original_answer=7.0,
                original_reasoning="2x + 3 = 17 => x = 7",
                constraint_role="constant",
                original_value=3,
                new_value=6,
            )

        assert isinstance(result, dict)
        assert result["new_answer"] == 10.0
        assert "new_reasoning" in result


@pytest.mark.asyncio
async def test_algebraic_uses_calibrated_prompt_when_available(monkeypatch):
    """When a calibration artifact exists, the councilor should use the DSPy program."""

    from elenchus.calibration import loader as loader_module

    mock_program = MagicMock()
    mock_program.return_value = MagicMock(
        answer=5.0,
        reasoning="DSPy optimized algebraic reasoning",
    )
    monkeypatch.setattr(loader_module, "load_optimized_prompt", lambda s, m: mock_program)

    import dspy

    class _DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(dspy, "LM", lambda model: object())
    monkeypatch.setattr(dspy, "context", lambda **kwargs: _DummyContext())

    councilor = AlgebraicCouncilor()
    result = await councilor.solve("Solve for x: 3x + 7 = 22")

    assert result.answer == 5.0
    assert result.strategy == "algebraic"
    mock_program.assert_called_once()
