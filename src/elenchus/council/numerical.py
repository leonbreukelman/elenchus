"""Numerical councilor — solves problems via estimate-then-verify strategy."""

from __future__ import annotations

from elenchus.council.base import BaseCouncilor


class NumericalCouncilor(BaseCouncilor):
    """Councilor that solves problems via estimate-then-verify."""

    strategy: str = "numerical"

    solve_prompt: str = """\
You are a precise mathematical solver using numerical estimation and verification.
Follow this two-step process:
1. ESTIMATE: Make a rough numerical estimate of the answer.
2. VERIFY: Work through the exact calculation to confirm or correct your estimate.

Return ONLY valid JSON with:
- "answer": the numeric answer (a number, not a string)
- "reasoning": your estimate-then-verify reasoning showing both steps
- "confidence": your confidence from 0.0 to 1.0

No markdown fences or extra text.
"""

    instruct_prompt: str = """\
You previously solved this problem using numerical estimation and verification:

Problem: {problem}
Your answer: {original_answer}
Your reasoning: {original_reasoning}

Now a constraint has changed. The {constraint_role} "{original_value}" \
has been changed to "{new_value}".

Using your same estimate-then-verify method, calculate the new answer step \
by step with this changed value:
1. ESTIMATE: What rough value do you expect with the new constraint?
2. CALCULATE: Work through the exact computation to get the precise answer.
Return ONLY valid JSON with:
- "new_answer": the new numeric answer (a number, not a string)
- "new_reasoning": the complete estimate-then-verify calculation

No markdown fences or extra text.
"""
