"""Centralised model configuration — loaded from environment variables.

Usage::

    from elenchus.config import get_model_config

    model = get_model_config().fast    # for router, judge, extractor
    model = get_model_config().capable # for councilors
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Which LLM models to use for each tier.

    Uses LiteLLM model strings (``"provider/model-name"``).
    All fields must be set via environment variables — there are no defaults.
    """

    fast: str
    """Cheap/fast model for routing, constraint extraction, mechanism judging."""

    capable: str
    """More capable model for core problem-solving (councilors, calibration)."""

    max_tokens_fast: int = 4096
    """Max completion tokens for the fast tier."""

    max_tokens_capable: int = 16384
    """Max completion tokens for the capable tier.

    Reasoning models (deepseek-r1, qwen3) routinely produce 8-23K tokens
    of chain-of-thought on math problems. 16384 provides headroom for most
    GSM8K-level problems; increase to 32768 for competition-level math.
    """


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    """Load model config from environment variables.

    Required env vars::

        ELENCHUS_MODEL_FAST=openrouter/qwen/qwen3-32b
        ELENCHUS_MODEL_CAPABLE=openrouter/deepseek/deepseek-r1-0528

    Optional::

        ELENCHUS_MAX_TOKENS_FAST=4096
        ELENCHUS_MAX_TOKENS_CAPABLE=16384
    """
    fast = os.getenv("ELENCHUS_MODEL_FAST")
    capable = os.getenv("ELENCHUS_MODEL_CAPABLE")

    if not fast or not capable:
        raise RuntimeError(
            "Model configuration required. Set environment variables:\n"
            "  ELENCHUS_MODEL_FAST=openrouter/qwen/qwen3-32b\n"
            "  ELENCHUS_MODEL_CAPABLE=openrouter/deepseek/deepseek-r1-0528\n"
            "\n"
            "Any LiteLLM model string works (openrouter/*, openai/*, etc.)"
        )

    return ModelConfig(
        fast=fast,
        capable=capable,
        max_tokens_fast=int(os.getenv("ELENCHUS_MAX_TOKENS_FAST", "4096")),
        max_tokens_capable=int(os.getenv("ELENCHUS_MAX_TOKENS_CAPABLE", "16384")),
    )
