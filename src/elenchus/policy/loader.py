"""Load domain policy configs from YAML with inheritance support.

Domain YAML files live in ``<project_root>/domains/``. A domain that
specifies ``extends: _base`` will deep-merge with the base config,
with the child's values taking precedence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from elenchus.policy.schemas import DomainConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base*, recursing into nested dicts.

    Lists and scalars in *override* replace those in *base* entirely.
    Only ``dict`` values are merged recursively.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_council(raw: dict) -> dict:
    """Flatten the YAML ``council.consensus`` sub-dict into the flat schema.

    The YAML files store council consensus settings as::

        council:
          consensus:
            tolerance_relative: 1.0e-6
            debate_rounds: 1

    The Pydantic ``CouncilConfig`` expects flat keys::

        consensus_tolerance_relative: 1.0e-6
        debate_rounds: 1

    This function bridges the two representations.
    """
    council = raw.get("council")
    if council is None or "consensus" not in council:
        return raw

    raw = raw.copy()
    council = council.copy()
    consensus = council.pop("consensus", {})

    for key, value in consensus.items():
        flat_key = f"consensus_{key}"
        council[flat_key] = value

    raw["council"] = council
    return raw


def _project_root() -> Path:
    """Resolve the project root (contains ``pyproject.toml``)."""
    here = Path(__file__).resolve().parent
    for ancestor in [here] + list(here.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here  # fallback


def load_domain_config(domain_name: str, domains_dir: Path | None = None) -> DomainConfig:
    """Load a domain config by name, merging with base if ``extends`` is set.

    Parameters
    ----------
    domain_name:
        Name of the domain (matches the YAML filename without extension).
    domains_dir:
        Directory containing the YAML files. Defaults to
        ``<project_root>/domains/``.

    Returns
    -------
    DomainConfig
        Fully resolved configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML file for *domain_name* does not exist.
    """
    if domains_dir is None:
        domains_dir = _project_root() / "domains"

    yaml_path = domains_dir / f"{domain_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Domain config not found: {yaml_path}")

    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text()) or {}

    # Resolve inheritance
    extends = raw.get("extends")
    if extends is not None:
        base_path = domains_dir / f"{extends}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Base domain config not found: {base_path}")
        base_raw: dict[str, Any] = yaml.safe_load(base_path.read_text()) or {}
        base_raw = _normalize_council(base_raw)
        raw = _normalize_council(raw)
        merged = _deep_merge(base_raw, raw)
    else:
        merged = _normalize_council(raw)

    # Ensure name is set
    if "name" not in merged:
        merged["name"] = domain_name

    return DomainConfig(**merged)
