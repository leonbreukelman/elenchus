"""DSPy prompt optimization runner for councilor calibration."""

from __future__ import annotations

from pathlib import Path

import dspy
import structlog

from elenchus.calibration.dataset import load_calibration_problems
from elenchus.calibration.signatures import (
    AlgebraicMathSolver,
    NumericalMathSolver,
)
from elenchus.config import get_model_config

logger = structlog.get_logger()

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

_STRATEGY_SIGNATURES = {
    "numerical": NumericalMathSolver,
    "algebraic": AlgebraicMathSolver,
}


def build_answer_metric(tolerance: float = 1e-2):
    """Build a DSPy metric that checks answer accuracy within tolerance.

    Returns a callable(example, prediction) -> float.
    """

    def metric(example, prediction, trace=None) -> float:
        try:
            expected = float(example.answer)
            predicted = float(prediction.answer)
        except (ValueError, TypeError, AttributeError):
            return 0.0

        if abs(expected) < 1e-10:
            return 1.0 if abs(predicted) < tolerance else 0.0

        rel_error = abs(predicted - expected) / abs(expected)
        return 1.0 if rel_error <= tolerance else 0.0

    return metric


def prepare_trainset(problems: list[dict]) -> list[dspy.Example]:
    """Convert calibration problems into DSPy Example objects."""
    examples = []
    for p in problems:
        ex = dspy.Example(
            problem=p["question"],
            answer=p["expected_answer"],
        ).with_inputs("problem")
        examples.append(ex)
    return examples


def run_optimization(
    strategy: str,
    model_name: str | None = None,
    num_trials: int = 10,
    num_candidates: int = 10,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 5,
) -> Path:
    """Run MIPROv2 optimization for a councilor strategy.

    Args:
        strategy: "numerical" or "algebraic"
        model_name: LiteLLM model string (e.g. "openrouter/deepseek/deepseek-r1-0528").
                    Defaults to ``get_model_config().capable``.
        num_trials: Number of MIPROv2 optimization trials
        max_bootstrapped_demos: Max few-shot examples from bootstrapping
        max_labeled_demos: Max labeled examples to include

    Returns:
        Path to the saved optimized program artifact.
    """
    if model_name is None:
        model_name = get_model_config().capable

    if strategy not in _STRATEGY_SIGNATURES:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {list(_STRATEGY_SIGNATURES)}")

    signature = _STRATEGY_SIGNATURES[strategy]
    problems = load_calibration_problems()
    trainset = prepare_trainset(problems)

    # Configure DSPy — model_name is already in LiteLLM format (e.g. "openrouter/...")
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)

    # Build the program
    program = dspy.ChainOfThought(signature)
    metric = build_answer_metric(tolerance=1e-2)

    # Run MIPROv2
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto=None,
        num_candidates=num_candidates,
        num_threads=4,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    logger.info(
        "calibration_starting",
        strategy=strategy,
        model=model_name,
        num_trials=num_trials,
        dataset_size=len(trainset),
    )

    optimized = optimizer.compile(
        program,
        trainset=trainset,
        num_trials=num_trials,
    )

    # Save artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    artifact_path = ARTIFACTS_DIR / f"{strategy}_{safe_model}.json"
    optimized.save(str(artifact_path))

    logger.info("calibration_complete", strategy=strategy, artifact=str(artifact_path))
    return artifact_path
