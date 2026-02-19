"""Base councilor with shared LLM-calling logic for solve and instruct."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from elenchus import extract_json, parse_number
from elenchus.config import get_model_config
from elenchus.llm import complete
from elenchus.state import CouncilorResult

log = structlog.get_logger()


class BaseCouncilor:
    """Base class that all councilor strategies extend.

    Subclasses MUST define:
        strategy: str          — e.g. "algebraic"
        solve_prompt: str      — system prompt for the solve call
        instruct_prompt: str    — template for the instruct call (with {problem}, etc.)

    Subclasses MAY override:
        _post_process_solve(data, problem) — custom post-processing of the parsed
            JSON from the solve call. Default: extract answer/reasoning/confidence.
    """

    strategy: str
    solve_prompt: str
    instruct_prompt: str

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    async def solve(self, problem: str) -> CouncilorResult:
        """Solve the given math problem — calibrated path first, then LLM fallback."""
        from elenchus.calibration import loader as loader_module

        model = get_model_config().capable

        # Try calibrated DSPy path first
        calibrated = loader_module.load_optimized_prompt(self.strategy, model)
        if calibrated is not None:
            try:
                import dspy

                dspy.configure(lm=dspy.LM(model))
                result = calibrated(problem=problem)
                answer = float(result.answer)
                log.info(f"{self.strategy}.solved_calibrated", answer=answer)
                return CouncilorResult(
                    strategy=self.strategy,
                    answer=answer,
                    reasoning=str(result.reasoning),
                    confidence=0.85,
                )
            except Exception:
                log.warning(f"{self.strategy}.calibrated_failed_fallback", exc_info=True)

        # LLM fallback via unified completion layer
        log.debug(f"{self.strategy}.solve", problem=problem[:80])

        response = await complete(
            model=model,
            messages=[{"role": "user", "content": problem}],
            system=self.solve_prompt,
            max_tokens=get_model_config().max_tokens_capable,
        )

        try:
            data = extract_json(response.text)
        except json.JSONDecodeError:
            # Regex fallback — try to extract answer from raw text
            m = re.search(
                r'"answer"\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)', response.text
            )
            if m:
                answer = float(m.group(1))
                log.warning(f"{self.strategy}.json_fallback_regex", answer=answer)
                return CouncilorResult(
                    strategy=self.strategy,
                    answer=answer,
                    reasoning="(JSON parse failed — answer extracted via regex)",
                    confidence=0.3,
                    code=None,
                )
            raise
        return self._post_process_solve(data, problem)

    def _post_process_solve(self, data: dict, problem: str) -> CouncilorResult:
        """Default post-processing: extract answer, reasoning, confidence from JSON.

        Override in subclasses that need custom parsing (e.g. SymbolicCouncilor).
        """
        result = CouncilorResult(
            strategy=self.strategy,
            answer=parse_number(data["answer"]),
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            code=None,
        )
        log.info(f"{self.strategy}.solved", answer=result.answer, confidence=result.confidence)
        return result

    # ------------------------------------------------------------------
    # instruct
    # ------------------------------------------------------------------
    async def instruct(
        self,
        problem: str,
        original_answer: Any,
        original_reasoning: str,
        constraint_role: str,
        original_value: Any,
        new_value: Any,
    ) -> dict:
        """Instruct the councilor to re-solve with a perturbed constraint."""
        model = get_model_config().capable

        prompt = self.instruct_prompt.format(
            problem=problem,
            original_answer=original_answer,
            original_reasoning=original_reasoning,
            constraint_role=constraint_role,
            original_value=original_value,
            new_value=new_value,
        )

        log.debug(f"{self.strategy}.instruct", constraint_role=constraint_role, new_value=new_value)

        response = await complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            system="You are a precise mathematical calculator. Calculate the answer step by step. Return only valid JSON.",
            max_tokens=get_model_config().max_tokens_capable,
        )

        try:
            return extract_json(response.text)
        except json.JSONDecodeError:
            m = re.search(
                r'"new_answer"\s*:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)',
                response.text,
            )
            if m:
                log.warning(f"{self.strategy}.instruct_json_fallback_regex")
                return {
                    "new_answer": float(m.group(1)),
                    "new_reasoning": "(JSON parse failed)",
                }
            raise
