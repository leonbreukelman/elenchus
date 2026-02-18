"""LLM judge for mechanism quality scoring in the Deutsch Probe."""

from __future__ import annotations

import structlog

from elenchus import extract_json
from elenchus.config import get_model_config
from elenchus.llm import complete

logger = structlog.get_logger()

JUDGE_PROMPT = """\
You are verifying whether a mathematical reasoning explanation justifies \
a specific numeric change.

A constraint was perturbed:
- What changed: {constraint_role}
- Original value: {original_value}
- New value: {new_value}

The answers:
- Original answer: {original_answer}
- Actual answer (ground truth): {actual_answer}
- Delta (actual change): {delta}

The councilor's reasoning about why the answer changed:
"{new_reasoning}"

Verify whether the stated reasoning mathematically justifies a delta of \
{delta}. Score 0.0-1.0:
1. Does the reasoning identify the correct mathematical relationship?
2. Is the explained mechanism consistent with a change of {delta}?
3. Is the explanation specific (names the relationship, not generic)?

Return ONLY valid JSON with:
- "score": float between 0.0 and 1.0
- "reasoning": one sentence explaining the score

No markdown fences or extra text.\
"""


async def judge_mechanism(
    constraint_role: str,
    original_value: float,
    new_value: float,
    original_answer: float,
    actual_answer: float,
    new_reasoning: str,
) -> float:
    """Score the quality of a councilor's mechanism explanation.

    Returns a float in [0.0, 1.0]. On any failure, returns 0.5 as a
    neutral fallback (same as the old hardcoded default).
    """
    if not new_reasoning or not new_reasoning.strip():
        logger.debug("mechanism_judge_no_reasoning")
        return 0.5

    delta = round(actual_answer - original_answer, 2)

    prompt = JUDGE_PROMPT.format(
        constraint_role=constraint_role,
        original_value=original_value,
        new_value=new_value,
        original_answer=original_answer,
        actual_answer=actual_answer,
        delta=delta,
        new_reasoning=new_reasoning,
    )

    try:
        model = get_model_config().fast

        response = await complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )

        data = extract_json(response.text)
        score = float(data["score"])
        logger.debug(
            "mechanism_judge_score",
            score=score,
            reasoning=data.get("reasoning", ""),
        )
        return max(0.0, min(1.0, score))
    except Exception:
        logger.warning("mechanism_judge_failed", exc_info=True)
        return 0.5
