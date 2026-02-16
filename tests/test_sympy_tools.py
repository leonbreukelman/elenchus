"""Tests for elenchus.tools.sympy_tools — answer comparison and code substitution."""

from __future__ import annotations

from elenchus.tools.sympy_tools import (
    answers_match_numeric,
    answers_match_symbolic,
    substitute_value_in_code,
)

# ---------------------------------------------------------------------------
# answers_match_numeric
# ---------------------------------------------------------------------------


class TestAnswersMatchNumeric:
    def test_exact_match(self):
        assert answers_match_numeric(3.0, 3.0) is True

    def test_within_tolerance(self):
        assert answers_match_numeric(3.0, 3.0000001) is True

    def test_mismatch(self):
        assert answers_match_numeric(3.0, 4.0) is False

    def test_zero_zero(self):
        assert answers_match_numeric(0.0, 0.0) is True

    def test_near_zero(self):
        """Very small values near zero use the denominator floor (1e-10)."""
        # Difference 1e-18 / floor 1e-10 = 1e-8, well within 1e-6 tolerance
        assert answers_match_numeric(1e-15, 1.001e-15) is True
        assert answers_match_numeric(0.0, 1e-17) is True
        # Far apart even with the floor
        assert answers_match_numeric(0.0, 1e-3) is False

    def test_negative_values(self):
        assert answers_match_numeric(-5.0, -5.0) is True
        assert answers_match_numeric(-5.0, 5.0) is False

    def test_custom_tolerance(self):
        assert answers_match_numeric(1.0, 1.1, rel_tol=0.2) is True
        assert answers_match_numeric(1.0, 1.1, rel_tol=0.05) is False


# ---------------------------------------------------------------------------
# answers_match_symbolic
# ---------------------------------------------------------------------------


class TestAnswersMatchSymbolic:
    def test_equivalent_expressions(self):
        assert answers_match_symbolic("x**2 + 2*x + 1", "(x + 1)**2") is True

    def test_different_expressions(self):
        assert answers_match_symbolic("x**2", "x**3") is False

    def test_numeric_equivalence(self):
        assert answers_match_symbolic("6", "2*3") is True

    def test_trig_identity(self):
        assert answers_match_symbolic("sin(x)**2 + cos(x)**2", "1") is True

    def test_symbolic_fraction(self):
        assert answers_match_symbolic("(x**2 - 1)/(x - 1)", "x + 1") is True


# ---------------------------------------------------------------------------
# substitute_value_in_code
# ---------------------------------------------------------------------------


class TestSubstituteValueInCode:
    def test_basic_substitution(self):
        code = "rate = 0.05\nresult = rate * 100"
        new_code = substitute_value_in_code(code, "rate", 0.08)
        assert "rate = 0.08" in new_code
        assert "0.05" not in new_code

    def test_preserves_other_lines(self):
        code = "x = 10\ny = 20\nz = x + y"
        new_code = substitute_value_in_code(code, "x", 42)
        assert "x = 42" in new_code
        assert "y = 20" in new_code
        assert "z = x + y" in new_code

    def test_integer_substitution(self):
        code = "n = 5"
        new_code = substitute_value_in_code(code, "n", 10)
        assert "n = 10" in new_code

    def test_handles_spaces_around_equals(self):
        code = "rate  =  0.05"
        new_code = substitute_value_in_code(code, "rate", 0.08)
        assert "0.08" in new_code
        assert "0.05" not in new_code

    def test_does_not_substitute_partial_name(self):
        """'rate' substitution must not affect 'interest_rate'."""
        code = "rate = 0.05\ninterest_rate = 0.03"
        new_code = substitute_value_in_code(code, "rate", 0.08)
        assert "interest_rate = 0.03" in new_code
