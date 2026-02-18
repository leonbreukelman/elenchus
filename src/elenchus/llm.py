"""Unified LLM completion layer — wraps LiteLLM for provider-agnostic calls.

All LLM interactions in Elenchus go through this module, making it the single
point of change when switching providers or adding retry/cost logic.

Handles reasoning/thinking models transparently: strips <think> blocks from
response text and exposes them separately via LLMResponse.reasoning_content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import litellm
import structlog

log = structlog.get_logger()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class UsageInfo:
    """Token usage and cost from a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""

    text: str
    model: str
    usage: UsageInfo = field(default_factory=UsageInfo)
    reasoning_content: str | None = None


def _strip_think_blocks(text: str) -> tuple[str, str | None]:
    """Strip <think>...</think> blocks from model output.

    Some reasoning models (deepseek-r1, qwen3 in thinking mode) embed
    chain-of-thought in <think> tags within the response content. LiteLLM
    usually separates these into reasoning_content, but some providers
    (OpenRouter) may leave them inline.

    Returns (cleaned_text, extracted_reasoning_or_None).
    """
    matches = _THINK_BLOCK_RE.findall(text)
    if not matches:
        return text, None
    reasoning = "\n".join(m.strip() for m in matches)
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    return cleaned, reasoning


async def complete(
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 16384,
    num_retries: int = 3,
) -> LLMResponse:
    """Call an LLM and return a typed response.

    Parameters
    ----------
    model:
        LiteLLM model string, e.g. ``"openrouter/qwen/qwen3-32b"``
        or ``"openrouter/deepseek/deepseek-r1-0528"``.
    messages:
        OpenAI-style message list.
    system:
        Optional system prompt.  Prepended as a system message.
    max_tokens:
        Maximum completion tokens.  Reasoning models need 16384+
        for math problems; see config.max_tokens_capable.
    num_retries:
        Number of retries on transient failures (LiteLLM built-in).
    """
    full_messages = list(messages)  # shallow copy
    if system:
        full_messages.insert(0, {"role": "system", "content": system})

    log.debug("llm.call", model=model, num_messages=len(full_messages))

    response = await litellm.acompletion(
        model=model,
        messages=full_messages,
        max_tokens=max_tokens,
        num_retries=num_retries,
    )

    raw_text = response.choices[0].message.content or ""
    usage_data = response.usage
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0  # Unknown model pricing

    # Extract reasoning content from LiteLLM's native field first
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)

    # Also strip any inline <think> blocks as a safety net
    text, inline_reasoning = _strip_think_blocks(raw_text)
    if not reasoning and inline_reasoning:
        reasoning = inline_reasoning

    usage = UsageInfo(
        prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
        completion_tokens=getattr(usage_data, "completion_tokens", 0),
        total_tokens=getattr(usage_data, "total_tokens", 0),
        cost_usd=cost,
    )

    log.debug(
        "llm.response",
        model=model,
        tokens=usage.total_tokens,
        cost_usd=f"${usage.cost_usd:.6f}",
        has_reasoning=reasoning is not None,
    )

    return LLMResponse(
        text=text,
        model=model,
        usage=usage,
        reasoning_content=reasoning,
    )
