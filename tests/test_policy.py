"""Tests for domain policy schemas and YAML loader."""

from pathlib import Path

import pytest
import yaml

from elenchus.policy.loader import load_domain_config
from elenchus.policy.schemas import (
    CouncilConfig,
    DomainConfig,
    ProbeConfig,
    RouterConfig,
    SandboxConfig,
)

# --- Schema defaults ---


class TestSchemaDefaults:
    """DomainConfig and sub-configs should have sensible defaults."""

    def test_sandbox_defaults(self):
        cfg = SandboxConfig()
        assert cfg.timeout == 30
        assert cfg.allowed_imports == ["sympy", "numpy"]

    def test_probe_defaults(self):
        cfg = ProbeConfig()
        assert cfg.sample_rate_unanimous == 0.30
        assert cfg.perturbation_budget == 3
        assert cfg.confidence_threshold == 0.80
        assert cfg.reject_threshold == 0.50
        assert isinstance(cfg.sandbox, SandboxConfig)
        assert cfg.preferred_perturbations == []

    def test_council_defaults(self):
        cfg = CouncilConfig()
        assert cfg.strategies == ["algebraic", "numerical", "symbolic"]
        assert cfg.consensus_tolerance_relative == pytest.approx(1e-6)
        assert cfg.debate_rounds == 1

    def test_router_defaults(self):
        cfg = RouterConfig()
        assert cfg.problem_types == []

    def test_domain_config_defaults(self):
        cfg = DomainConfig()
        assert cfg.name == "_base"
        assert cfg.extends is None
        assert isinstance(cfg.probe, ProbeConfig)
        assert isinstance(cfg.council, CouncilConfig)
        assert isinstance(cfg.router, RouterConfig)


# --- YAML loader ---


class TestLoadDomainConfig:
    """Loader should read YAML, apply defaults, and resolve inheritance."""

    def _write_yaml(self, path: Path, data: dict) -> None:
        path.write_text(yaml.dump(data, default_flow_style=False))

    def test_base_config_loads(self, tmp_path: Path):
        base_data = {
            "probe": {
                "sample_rate_unanimous": 0.25,
                "perturbation_budget": 5,
            },
        }
        self._write_yaml(tmp_path / "_base.yaml", base_data)

        cfg = load_domain_config("_base", domains_dir=tmp_path)

        assert cfg.name == "_base"
        assert cfg.probe.sample_rate_unanimous == 0.25
        assert cfg.probe.perturbation_budget == 5
        # Non-overridden defaults survive
        assert cfg.probe.confidence_threshold == 0.80

    def test_domain_extends_base(self, tmp_path: Path):
        base_data = {
            "probe": {
                "sample_rate_unanimous": 0.25,
                "perturbation_budget": 5,
            },
            "council": {
                "strategies": ["algebraic", "numerical", "symbolic"],
            },
        }
        child_data = {
            "name": "algebra",
            "extends": "_base",
            "router": {
                "problem_types": ["equation", "system"],
            },
            "probe": {
                "perturbation_budget": 10,
            },
        }
        self._write_yaml(tmp_path / "_base.yaml", base_data)
        self._write_yaml(tmp_path / "algebra.yaml", child_data)

        cfg = load_domain_config("algebra", domains_dir=tmp_path)

        assert cfg.name == "algebra"
        assert cfg.extends == "_base"
        # Inherited from base
        assert cfg.probe.sample_rate_unanimous == 0.25
        assert cfg.council.strategies == ["algebraic", "numerical", "symbolic"]
        # Overridden by child
        assert cfg.probe.perturbation_budget == 10
        assert cfg.router.problem_types == ["equation", "system"]

    def test_missing_domain_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_domain_config("nonexistent", domains_dir=tmp_path)

    def test_deep_merge_nested_dicts(self, tmp_path: Path):
        base_data = {
            "probe": {
                "sandbox": {
                    "timeout": 60,
                    "allowed_imports": ["sympy"],
                },
            },
        }
        child_data = {
            "name": "custom",
            "extends": "_base",
            "probe": {
                "sandbox": {
                    "timeout": 120,
                },
            },
        }
        self._write_yaml(tmp_path / "_base.yaml", base_data)
        self._write_yaml(tmp_path / "custom.yaml", child_data)

        cfg = load_domain_config("custom", domains_dir=tmp_path)

        # Child overrides timeout
        assert cfg.probe.sandbox.timeout == 120
        # Base's allowed_imports preserved through deep merge
        assert cfg.probe.sandbox.allowed_imports == ["sympy"]

    def test_real_domain_files(self):
        """Smoke test against the actual domain YAML files in the repo."""
        domains_dir = Path(__file__).resolve().parent.parent / "domains"
        if not (domains_dir / "_base.yaml").exists():
            pytest.skip("Real domain YAML files not present")

        base = load_domain_config("_base", domains_dir=domains_dir)
        assert base.name == "_base"

        algebra = load_domain_config("algebra", domains_dir=domains_dir)
        assert algebra.name == "algebra"
        assert algebra.extends == "_base"
        assert "equation" in algebra.router.problem_types
