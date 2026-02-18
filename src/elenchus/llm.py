"""Unified LLM completion layer — wraps LiteLLM for provider-agnostic calls.

All LLM interactions in Elenchus go through this module, making it the single
point of change when switching providers or adding retry/cost logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import litellm
import structlog

log = structlog.get_logger()


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


async def complete(
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 8192,
    num_retries: int = 3,
) -> LLMResponse:
    """Call an LLM and return a typed response.

    Parameters
    ----------
    model:
        LiteLLM model string, e.g. ``"anthropic/claude-haiku-4-5-20251001"``
        or ``"openai/gpt-4o"``.
    messages:
        OpenAI-style message list.
    system:
        Optional system prompt.  Prepended as a system message.
    max_tokens:
        Maximum completion tokens.
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

    text = response.choices[0].message.content or ""
    usage_data = response.usage
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0  # Unknown model pricing (e.g. CLIProxyAPI)

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
    )

    return LLMResponse(
        text=text,
        model=model,
        usage=usage,
    )
