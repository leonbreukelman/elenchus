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
    """

    fast: str = "anthropic/claude-haiku-4-5-20251001"
    """Cheap/fast model for routing, constraint extraction, mechanism judging."""

    capable: str = "anthropic/claude-sonnet-4-5-20250929"
    """More capable model for core problem-solving (councilors, calibration)."""


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    """Load model config from environment with sensible Anthropic defaults.

    Override via::

        ELENCHUS_MODEL_FAST=openai/gpt-4o-mini
        ELENCHUS_MODEL_CAPABLE=openai/gpt-4o
    """
    return ModelConfig(
        fast=os.getenv("ELENCHUS_MODEL_FAST", ModelConfig.model_fields["fast"].default),
        capable=os.getenv("ELENCHUS_MODEL_CAPABLE", ModelConfig.model_fields["capable"].default),
    )
