"""LLM judge for mechanism quality scoring in the Deutsch Probe."""

from __future__ import annotations

import anthropic
import structlog

from elenchus import extract_json

logger = structlog.get_logger()

JUDGE_MODEL = "claude-haiku-4-5-20251001"

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
"{predicted_reasoning}"

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


def _get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


async def judge_mechanism(
    constraint_role: str,
    original_value: float,
    new_value: float,
    original_answer: float,
    actual_answer: float,
    predicted_reasoning: str,
) -> float:
    """Score the quality of a councilor's mechanism explanation.

    Returns a float in [0.0, 1.0]. On any failure, returns 0.5 as a
    neutral fallback (same as the old hardcoded default).
    """
    if not predicted_reasoning or not predicted_reasoning.strip():
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
        predicted_reasoning=predicted_reasoning,
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        data = extract_json(raw)
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
