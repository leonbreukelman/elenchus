"""Load optimized DSPy prompt artifacts at runtime."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_optimized_prompt(strategy: str, model_name: str):
    """Load an optimized DSPy program for a councilor strategy + model.

    Returns the loaded program, or None if no artifact exists.
    """
    safe_model = model_name.replace("/", "_")
    artifact_path = ARTIFACTS_DIR / f"{strategy}_{safe_model}.json"

    if not artifact_path.exists():
        logger.debug("no_calibration_artifact", strategy=strategy, model=model_name)
        return None

    try:
        import dspy

        from elenchus.calibration.signatures import (
            AlgebraicMathSolver,
            NumericalMathSolver,
        )

        sig_map = {
            "numerical": NumericalMathSolver,
            "algebraic": AlgebraicMathSolver,
        }
        sig = sig_map.get(strategy)
        if not sig:
            logger.warning("unknown_strategy_for_calibration", strategy=strategy)
            return None

        program = dspy.ChainOfThought(sig)
        program.load(str(artifact_path))
        logger.info("calibration_artifact_loaded", strategy=strategy, model=model_name)
        return program
    except Exception:
        logger.warning("calibration_load_failed", strategy=strategy, exc_info=True)
        return None
