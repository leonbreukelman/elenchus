"""Algebraic councilor — solves problems via step-by-step algebraic reasoning."""

from __future__ import annotations

from elenchus.council.base import BaseCouncilor


class AlgebraicCouncilor(BaseCouncilor):
    """Councilor that solves problems via algebraic manipulation."""

    strategy: str = "algebraic"

    solve_prompt: str = """\
You are a precise mathematical solver using algebraic methods.
Solve the given problem step by step using algebraic manipulation.

Return ONLY valid JSON with:
- "answer": the numeric answer (a number, not a string)
- "reasoning": step-by-step algebraic solution
- "confidence": your confidence from 0.0 to 1.0

No markdown fences or extra text.
"""

    instruct_prompt: str = """\
You previously solved this problem using algebraic methods:

Problem: {problem}
Your answer: {original_answer}
Your reasoning: {original_reasoning}

Now a constraint has changed. The {constraint_role} "{original_value}" \
has been changed to "{new_value}".

Using your same algebraic method, calculate the new answer step by step \
with this changed value. Show each step of the derivation.
Return ONLY valid JSON with:
- "new_answer": the new numeric answer (a number, not a string)
- "new_reasoning": the complete step-by-step calculation with the new value

No markdown fences or extra text.
"""
