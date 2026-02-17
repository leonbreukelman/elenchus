"""Tests for calibration artifact loading."""

from unittest.mock import patch


def test_load_optimized_prompt_returns_none_when_no_artifact():
    """When no artifact exists, should return None (fall back to hand-written)."""
    from elenchus.calibration.loader import load_optimized_prompt

    result = load_optimized_prompt("numerical", "nonexistent-model")
    assert result is None


def test_load_optimized_prompt_returns_program_when_artifact_exists(tmp_path):
    """When an artifact exists, should load and return the DSPy program."""
    import dspy

    from elenchus.calibration.loader import load_optimized_prompt
    from elenchus.calibration.signatures import NumericalMathSolver

    # Create a fake artifact
    program = dspy.ChainOfThought(NumericalMathSolver)
    artifact_path = tmp_path / "numerical_test-model.json"
    program.save(str(artifact_path))

    with patch("elenchus.calibration.loader.ARTIFACTS_DIR", tmp_path):
        result = load_optimized_prompt("numerical", "test-model")

    assert result is not None
