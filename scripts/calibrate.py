#!/usr/bin/env python
"""CLI entry point for running DSPy prompt calibration.

Usage:
    uv run python scripts/calibrate.py --strategy numerical
    uv run python scripts/calibrate.py --strategy algebraic --model claude-sonnet-4-5-20250929
    uv run python scripts/calibrate.py --strategy numerical --trials 20
"""

from __future__ import annotations

import argparse

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
)


def main():
    parser = argparse.ArgumentParser(description="Run DSPy prompt calibration for Elenchus councilors")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["numerical", "algebraic"],
        help="Which councilor strategy to calibrate",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model ID to optimize for (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of MIPROv2 optimization trials (default: 10)",
    )
    args = parser.parse_args()

    from elenchus.calibration.optimize import run_optimization

    artifact_path = run_optimization(
        strategy=args.strategy,
        model_name=args.model,
        num_trials=args.trials,
    )
    print(f"\nCalibration complete. Artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
