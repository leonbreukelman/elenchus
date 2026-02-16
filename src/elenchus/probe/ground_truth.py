"""Phase 4: Compute symbolic ground truth for perturbations."""

from __future__ import annotations

import structlog

from elenchus.state import Perturbation
from elenchus.tools.sandbox import SandboxResult, execute_code
from elenchus.tools.sympy_tools import substitute_value_in_code

logger = structlog.get_logger()


async def compute_ground_truth(
    original_code: str,
    perturbation: Perturbation,
    timeout: int = 30,
) -> SandboxResult:
    """Substitute perturbed value into code and execute."""
    perturbed_code = substitute_value_in_code(
        original_code,
        perturbation.constraint.name,
        perturbation.new_value,
    )
    logger.info(
        "ground_truth_compute",
        constraint=perturbation.constraint.name,
        new_value=perturbation.new_value,
    )
    return await execute_code(perturbed_code, timeout=timeout, allowed_imports=["sympy"])
