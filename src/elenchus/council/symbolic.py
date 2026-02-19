"""Symbolic councilor — generates SymPy code and executes it in a sandbox."""

from __future__ import annotations

import re

import structlog

from elenchus import parse_number
from elenchus.config import get_model_config
from elenchus.council.base import BaseCouncilor
from elenchus.llm import complete
from elenchus.state import CouncilorResult
from elenchus.tools.sandbox import execute_code

log = structlog.get_logger()


def _parse_numeric_output(output: str) -> float:
    """Extract a numeric answer from sandbox output.

    Tries in order:
    1. Last line as a plain float
    2. Last number found anywhere in the output
    """
    lines = output.strip().splitlines()
    last_line = lines[-1].strip()
    try:
        return parse_number(last_line)
    except ValueError:
        pass
    # Find all numbers in the output, return the last one
    numbers = re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", output)
    if numbers:
        return float(numbers[-1])
    raise ValueError(f"No numeric value found in output: {output!r}")


class SymbolicCouncilor(BaseCouncilor):
    """Councilor that generates SymPy code and executes it in a sandbox."""

    strategy: str = "symbolic"

    solve_prompt: str = """\
You are a mathematical solver that generates SymPy Python code.
Given a math problem, write Python code using SymPy to solve it.
The code MUST print the final numeric answer as the last line of output.

IMPORTANT: Use ONLY English ASCII variable names (e.g. price, count, rate). \
Never use non-ASCII characters in variable names.

IMPORTANT: Assign every given numeric value to a named variable on its own line \
BEFORE using it in equations. For example, write:
  h0 = 20
  v0 = 15
  equation = h0 + v0*t
NOT:
  equation = 20 + 15*t

Return ONLY valid JSON with:
- "code": the complete Python code (using sympy, must print the answer)
- "reasoning": brief explanation of the approach

No markdown fences or extra text.
"""

    instruct_prompt: str = """\
You previously solved this problem by generating and executing SymPy code:

Problem: {problem}
Your answer: {original_answer}
Your reasoning: {original_reasoning}

Now a constraint has changed. The {constraint_role} "{original_value}" \
has been changed to "{new_value}".

Using your understanding of the mathematical structure, calculate the new \
answer step by step with this changed value. Show the complete derivation.
Return ONLY valid JSON with:
- "new_answer": the new numeric answer (a number, not a string)
- "new_reasoning": the complete step-by-step calculation with the new value

No markdown fences or extra text.
"""

    async def solve(self, problem: str) -> CouncilorResult:
        """Generate SymPy code, execute in sandbox, return parsed result.

        Overrides BaseCouncilor.solve() because the symbolic councilor needs
        to execute code rather than just parsing JSON from the LLM response.
        """
        model = get_model_config().capable

        log.debug("symbolic.solve", problem=problem[:80])

        response = await complete(
            model=model,
            messages=[{"role": "user", "content": problem}],
            system=self.solve_prompt,
        )

        from elenchus import extract_json

        data = extract_json(response.text)
        code = data["code"]
        reasoning = data["reasoning"]

        log.debug("symbolic.executing_code", code_len=len(code))
        sandbox_result = await execute_code(code)

        if sandbox_result.success:
            # Try clean parse first (last line is a plain number)
            lines = sandbox_result.output.strip().splitlines()
            last_line = lines[-1].strip() if lines else ""
            try:
                answer = parse_number(last_line)
                confidence = 0.95  # High — clean sandbox output
            except ValueError:
                # Fallback: regex extraction — less certain
                try:
                    answer = _parse_numeric_output(sandbox_result.output)
                    confidence = 0.70  # Regex fallback — lower confidence
                except (ValueError, IndexError):
                    log.warning("symbolic.parse_failed", output=sandbox_result.output[:200])
                    return CouncilorResult(
                        strategy=self.strategy,
                        answer=None,
                        reasoning=reasoning,
                        confidence=0.0,
                        code=code,
                    )

            log.info("symbolic.solved", answer=answer, confidence=confidence)
            return CouncilorResult(
                strategy=self.strategy,
                answer=answer,
                reasoning=reasoning,
                confidence=confidence,
                code=code,
            )

        log.warning("symbolic.sandbox_failed", error=sandbox_result.error[:200])
        return CouncilorResult(
            strategy=self.strategy,
            answer=None,
            reasoning=reasoning,
            confidence=0.0,
            code=code,
        )
