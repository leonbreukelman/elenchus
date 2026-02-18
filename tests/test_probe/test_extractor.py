"""Tests for constraint extraction from problems."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from elenchus.llm import LLMResponse, UsageInfo
from elenchus.probe.extractor import extract_constraints
from elenchus.state import Constraint


def _make_llm_response(data) -> LLMResponse:
    return LLMResponse(text=json.dumps(data), model="test", usage=UsageInfo())


@pytest.mark.asyncio
async def test_extract_constraints_compound_interest():
    mock_response = _make_llm_response(
        [
            {"name": "principal", "original_value": 10000, "dtype": "numeric",
             "role": "initial investment amount", "perturbation_range": [1000, 50000]},
            {"name": "rate", "original_value": 0.05, "dtype": "numeric",
             "role": "annual interest rate", "perturbation_range": [0.01, 0.20]},
            {"name": "time", "original_value": 3, "dtype": "numeric",
             "role": "investment period in years", "perturbation_range": [1, 30]},
        ]
    )

    with patch("elenchus.probe.extractor.complete", new_callable=AsyncMock, return_value=mock_response):
        constraints = await extract_constraints(
            problem="$10,000 at 5% compounded monthly for 3 years",
            solution="$11,614.72",
        )

    assert len(constraints) == 3
    assert all(isinstance(c, Constraint) for c in constraints)
    assert constraints[0].name == "principal"
    assert constraints[1].perturbation_range == (0.01, 0.20)


@pytest.mark.asyncio
async def test_extract_constraints_with_symbolic_code():
    """When symbolic_code is provided, constraint names must match code variable names."""
    symbolic_code = (
        "from sympy import *\n"
        "P = 10000\n"
        "r = Rational(5, 100)\n"
        "n = 12\n"
        "t = 3\n"
        "A = P * (1 + r/n) ** (n*t)\n"
        "print(float(A))\n"
    )
    mock_response = _make_llm_response(
        [
            {"name": "P", "original_value": 10000, "dtype": "numeric",
             "role": "principal amount", "perturbation_range": [1000, 50000]},
            {"name": "r", "original_value": 0.05, "dtype": "numeric",
             "role": "annual interest rate", "perturbation_range": [0.01, 0.20]},
            {"name": "n", "original_value": 12, "dtype": "numeric",
             "role": "compounding frequency per year", "perturbation_range": [1, 365]},
            {"name": "t", "original_value": 3, "dtype": "numeric",
             "role": "investment period in years", "perturbation_range": [1, 30]},
        ]
    )

    with patch("elenchus.probe.extractor.complete", new_callable=AsyncMock, return_value=mock_response) as mock_complete:
        constraints = await extract_constraints(
            problem="$10,000 at 5% compounded monthly for 3 years",
            solution="$11,614.72",
            symbolic_code=symbolic_code,
        )

    assert len(constraints) == 4
    names = [c.name for c in constraints]
    assert names == ["P", "r", "n", "t"]

    # Verify the prompt included the symbolic code
    call_args = mock_complete.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "P = 10000" in prompt_content
    assert "variable names from the code" in prompt_content


@pytest.mark.asyncio
async def test_extract_constraints_without_symbolic_code_uses_fallback():
    """When symbolic_code is None, the original prompt is used (no code reference)."""
    mock_response = _make_llm_response(
        [
            {"name": "principal", "original_value": 10000, "dtype": "numeric",
             "role": "initial investment amount", "perturbation_range": [1000, 50000]},
        ]
    )

    with patch("elenchus.probe.extractor.complete", new_callable=AsyncMock, return_value=mock_response) as mock_complete:
        constraints = await extract_constraints(
            problem="$10,000 at 5% compounded monthly for 3 years",
            solution="$11,614.72",
            symbolic_code=None,
        )

    assert len(constraints) == 1
    assert constraints[0].name == "principal"

    # Verify the prompt did NOT include code-specific instructions
    call_args = mock_complete.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "variable names from the code" not in prompt_content
    assert "Code:" not in prompt_content
