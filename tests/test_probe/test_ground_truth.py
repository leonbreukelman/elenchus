from unittest.mock import patch

import pytest

from elenchus.probe.ground_truth import compute_ground_truth
from elenchus.state import Constraint, Perturbation
from elenchus.tools.sandbox import SandboxResult


@pytest.mark.asyncio
async def test_ground_truth_substitution():
    code = "rate = 0.05\nprincipal = 10000\nresult = principal * (1 + rate)**3\nprint(result)"
    constraint = Constraint(
        name="rate",
        original_value=0.05,
        dtype="numeric",
        role="annual rate",
        perturbation_range=(0.01, 0.20),
    )
    perturbation = Perturbation(constraint=constraint, new_value=0.08, rationale="test")

    mock_result = SandboxResult(success=True, output="12597.12", error="")

    with patch("elenchus.probe.ground_truth.execute_code", return_value=mock_result):
        result = await compute_ground_truth(code, perturbation)

    assert result.success is True
    assert float(result.output) == 12597.12


@pytest.mark.asyncio
async def test_ground_truth_sandbox_failure():
    code = "rate = 0.05\nbad code"
    constraint = Constraint(
        name="rate",
        original_value=0.05,
        dtype="numeric",
        role="rate",
        perturbation_range=(0.01, 0.20),
    )
    perturbation = Perturbation(constraint=constraint, new_value=0.08, rationale="test")

    mock_result = SandboxResult(success=False, output="", error="SyntaxError")

    with patch("elenchus.probe.ground_truth.execute_code", return_value=mock_result):
        result = await compute_ground_truth(code, perturbation)

    assert result.success is False
