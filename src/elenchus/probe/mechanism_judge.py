"""LLM judge for mechanism quality scoring in the Deutsch Probe."""

from __future__ import annotations

import anthropic
import structlog

from elenchus import extract_json

logger = structlog.get_logger()

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_PROMPT = """\
You are evaluating whether a mathematical reasoning explanation is coherent \
and consistent with the observed change in answer.

A constraint was perturbed:
- What changed: {constraint_role}
- Original value: {original_value}
- New value: {new_value}

The answers:
- Original answer: {original_answer}
- Actual answer (ground truth): {actual_answer}

The councilor's reasoning about why the answer changed:
"{predicted_reasoning}"

Score 0.0-1.0 on whether the stated mechanism is:
1. Mathematically coherent (the reasoning makes mathematical sense)
2. Consistent with the observed change (the direction and magnitude of the \
explanation matches what actually happened)
3. Specific (names the actual mathematical relationship, not generic hand-waving)

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

    prompt = JUDGE_PROMPT.format(
        constraint_role=constraint_role,
        original_value=original_value,
        new_value=new_value,
        original_answer=original_answer,
        actual_answer=actual_answer,
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
