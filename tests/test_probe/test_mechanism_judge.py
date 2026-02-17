"""Tests for the LLM mechanism judge."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_judge_returns_score_between_0_and_1():
    """The judge should return a float score in [0.0, 1.0]."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            text='{"score": 0.8, "reasoning": "The explanation correctly identifies the exponential relationship."}'
        )
    ]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("elenchus.probe.mechanism_judge._get_client", return_value=mock_client):
        score = await judge_mechanism(
            constraint_role="annual interest rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=11614.72,
            actual_answer=12682.42,
            predicted_reasoning="The rate is in the exponent base, so increasing it compounds more aggressively.",
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
        predicted_reasoning="",
    )
    assert score == 0.5


@pytest.mark.asyncio
async def test_judge_returns_fallback_on_api_failure():
    """API failure should return 0.5 fallback, not crash."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    with patch(
        "elenchus.probe.mechanism_judge._get_client",
        side_effect=Exception("API down"),
    ):
        score = await judge_mechanism(
            constraint_role="rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=100.0,
            actual_answer=120.0,
            predicted_reasoning="Rate affects compounding.",
        )
    assert score == 0.5


@pytest.mark.asyncio
async def test_judge_clamps_out_of_range_scores():
    """Scores outside [0, 1] should be clamped."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"score": 1.5, "reasoning": "Great"}')]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("elenchus.probe.mechanism_judge._get_client", return_value=mock_client):
        score = await judge_mechanism(
            constraint_role="rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=100.0,
            actual_answer=120.0,
            predicted_reasoning="Higher rate means more growth.",
        )
    assert score == 1.0
