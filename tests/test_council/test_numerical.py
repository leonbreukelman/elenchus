"""Tests for elenchus.council.numerical — numerical councilor strategy."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elenchus.council.base import BaseCouncilor
from elenchus.council.numerical import NumericalCouncilor
from elenchus.llm import LLMResponse, UsageInfo
from elenchus.state import CouncilorResult


def _make_llm_response(data: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(data), model="test", usage=UsageInfo())


# ---------------------------------------------------------------------------
# NumericalCouncilor
# ---------------------------------------------------------------------------


class TestNumericalCouncilor:
    def test_is_base_councilor(self):
        assert issubclass(NumericalCouncilor, BaseCouncilor)

    def test_strategy(self):
        c = NumericalCouncilor()
        assert c.strategy == "numerical"

    async def test_solve(self):
        mock_response = _make_llm_response(
            {
                "answer": 42.0,
                "reasoning": "Estimate: ~40. Verify: 6 * 7 = 42. Confirmed.",
                "confidence": 0.9,
            }
        )

        with patch("elenchus.council.base.complete", new_callable=AsyncMock, return_value=mock_response):
            councilor = NumericalCouncilor()
            result = await councilor.solve("What is 6 times 7?")

        assert isinstance(result, CouncilorResult)
        assert result.strategy == "numerical"
        assert result.answer == 42.0
        assert "Estimate" in result.reasoning
        assert result.confidence == 0.9
        assert result.code is None

    async def test_instruct(self):
        mock_response = _make_llm_response(
            {
                "new_answer": 48.0,
                "new_reasoning": "Changing 7 to 8: 6 * 8 = 48",
            }
        )

        with patch("elenchus.council.base.complete", new_callable=AsyncMock, return_value=mock_response):
            councilor = NumericalCouncilor()
            result = await councilor.instruct(
                problem="What is 6 times 7?",
                original_answer=42.0,
                original_reasoning="6 * 7 = 42",
                constraint_role="multiplicand",
                original_value=7,
                new_value=8,
            )

        assert isinstance(result, dict)
        assert result["new_answer"] == 48.0
        assert "new_reasoning" in result


@pytest.mark.asyncio
async def test_numerical_uses_calibrated_prompt_when_available(monkeypatch):
    """When a calibration artifact exists, the councilor should use the DSPy program."""

    from elenchus.calibration import loader as loader_module

    mock_program = MagicMock()
    mock_program.return_value = MagicMock(
        answer=42.0,
        reasoning="DSPy optimized reasoning",
    )
    monkeypatch.setattr(loader_module, "load_optimized_prompt", lambda s, m: mock_program)

    import dspy

    monkeypatch.setattr(dspy, "configure", lambda **kwargs: None)

    councilor = NumericalCouncilor()
    result = await councilor.solve("What is 6 * 7?")

    assert result.answer == 42.0
    assert result.strategy == "numerical"
    mock_program.assert_called_once()
