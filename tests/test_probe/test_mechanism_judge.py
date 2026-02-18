"""Tests for the LLM mechanism judge."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from elenchus.llm import LLMResponse, UsageInfo


def _make_llm_response(data: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(data), model="test", usage=UsageInfo())


@pytest.mark.asyncio
async def test_judge_returns_score_between_0_and_1():
    """The judge should return a float score in [0.0, 1.0]."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    mock_response = _make_llm_response(
        {"score": 0.8, "reasoning": "The explanation correctly identifies the exponential relationship."}
    )

    with patch("elenchus.probe.mechanism_judge.complete", new_callable=AsyncMock, return_value=mock_response):
        score = await judge_mechanism(
            constraint_role="annual interest rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=11614.72,
            actual_answer=12682.42,
            new_reasoning="The rate is in the exponent base, so increasing it compounds more aggressively.",
        )

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_judge_returns_fallback_on_empty_reasoning():
    """Empty reasoning should return 0.5 without making an API call."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    score = await judge_mechanism(
        constraint_role="rate",
        original_value=0.05,
        new_value=0.08,
        original_answer=100.0,
        actual_answer=120.0,
        new_reasoning="",
    )
    assert score == 0.5


@pytest.mark.asyncio
async def test_judge_returns_fallback_on_api_failure():
    """API failure should return 0.5 fallback, not crash."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    with patch(
        "elenchus.probe.mechanism_judge.complete",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ):
        score = await judge_mechanism(
            constraint_role="rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=100.0,
            actual_answer=120.0,
            new_reasoning="Rate affects compounding.",
        )
    assert score == 0.5


@pytest.mark.asyncio
async def test_judge_clamps_out_of_range_scores():
    """Scores outside [0, 1] should be clamped."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    mock_response = _make_llm_response({"score": 1.5, "reasoning": "Great"})

    with patch("elenchus.probe.mechanism_judge.complete", new_callable=AsyncMock, return_value=mock_response):
        score = await judge_mechanism(
            constraint_role="rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=100.0,
            actual_answer=120.0,
            new_reasoning="Higher rate means more growth.",
        )
    assert score == 1.0


@pytest.mark.asyncio
async def test_judge_prompt_includes_delta_information():
    """The judge prompt should include the numeric delta for verification."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    mock_response = _make_llm_response({"score": 0.9, "reasoning": "Delta is correct."})

    with patch("elenchus.probe.mechanism_judge.complete", new_callable=AsyncMock, return_value=mock_response) as mock_complete:
        await judge_mechanism(
            constraint_role="annual interest rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=11614.72,
            actual_answer=12682.42,
            new_reasoning="Higher rate increases compounding.",
        )

    # Inspect the prompt sent to the model
    call_args = mock_complete.call_args
    prompt_text = call_args.kwargs["messages"][0]["content"]

    # Must contain the computed delta
    assert "1067.7" in prompt_text  # 12682.42 - 11614.72 = 1067.70
    # Must frame as verification, not open evaluation
    assert "verify" in prompt_text.lower() or "justify" in prompt_text.lower()
