"""SymPy-based utilities for answer comparison and code substitution.

Used by the consensus engine (answer comparison) and Deutsch Probe Phase 4
(code substitution for ground truth computation).
"""

from __future__ import annotations

import re

import sympy


def answers_match_numeric(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    """Check if two numeric answers match within relative tolerance.

    Uses a denominator floor of 1e-10 to handle near-zero comparisons
    gracefully.
    """
    if a == b:
        return True
    denominator = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denominator < rel_tol


def answers_match_symbolic(expr_a: str, expr_b: str) -> bool:
    """Check if two symbolic expressions are mathematically equivalent.

    Parses both strings with ``sympy.sympify``, computes their difference,
    and simplifies.  Returns ``True`` when the simplified difference is zero.
    """
    a = sympy.sympify(expr_a)
    b = sympy.sympify(expr_b)
    return sympy.simplify(a - b) == 0


def substitute_value_in_code(code: str, variable_name: str, new_value: float | int) -> str:
    """Replace an assignment like ``var = old_value`` with ``var = new_value`` in code.

    Only matches assignments where *variable_name* appears as a whole word at
    the start of a line (after optional whitespace), so ``rate`` won't match
    ``interest_rate``.
    """
    pattern = re.compile(
        rf"^(\s*{re.escape(variable_name)}\s*=\s*)(.+)$",
        re.MULTILINE,
    )
    return pattern.sub(rf"\g<1>{new_value}", code)
