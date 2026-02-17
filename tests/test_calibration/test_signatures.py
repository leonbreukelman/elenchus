"""Tests for DSPy signatures."""


def test_math_solver_signature_has_required_fields():
    """The solver signature must have problem input and answer/reasoning output."""
    from elenchus.calibration.signatures import MathSolver

    assert "problem" in MathSolver.model_fields
    assert "answer" in MathSolver.model_fields
    assert "reasoning" in MathSolver.model_fields


def test_numerical_solver_signature_exists():
    """Numerical solver should have a distinct signature."""
    from elenchus.calibration.signatures import NumericalMathSolver

    assert "problem" in NumericalMathSolver.model_fields
    assert "answer" in NumericalMathSolver.model_fields


def test_algebraic_solver_signature_exists():
    """Algebraic solver should have a distinct signature."""
    from elenchus.calibration.signatures import AlgebraicMathSolver

    assert "problem" in AlgebraicMathSolver.model_fields
    assert "answer" in AlgebraicMathSolver.model_fields
