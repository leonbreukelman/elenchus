"""Tests for the DSPy optimization runner."""

from unittest.mock import MagicMock


def test_build_metric_returns_callable():
    """The metric factory should return a callable that scores answer accuracy."""
    from elenchus.calibration.optimize import build_answer_metric

    metric = build_answer_metric(tolerance=1e-2)
    assert callable(metric)


def test_metric_scores_correct_answer():
    """A correct answer (within tolerance) should score 1.0."""
    from elenchus.calibration.optimize import build_answer_metric

    metric = build_answer_metric(tolerance=1e-2)

    example = MagicMock()
    example.answer = 42.0

    prediction = MagicMock()
    prediction.answer = 42.1  # Within 1% tolerance

    score = metric(example, prediction)
    assert score == 1.0


def test_metric_scores_wrong_answer():
    """A wrong answer (outside tolerance) should score 0.0."""
    from elenchus.calibration.optimize import build_answer_metric

    metric = build_answer_metric(tolerance=1e-2)

    example = MagicMock()
    example.answer = 42.0

    prediction = MagicMock()
    prediction.answer = 50.0  # 19% off — way outside tolerance

    score = metric(example, prediction)
    assert score == 0.0


def test_prepare_trainset_creates_dspy_examples():
    """Should convert our problem dicts into DSPy Example objects."""
    from elenchus.calibration.optimize import prepare_trainset

    problems = [
        {"question": "2+2?", "expected_answer": 4.0, "category": "equation"},
        {"question": "3*3?", "expected_answer": 9.0, "category": "equation"},
    ]
    trainset = prepare_trainset(problems)
    assert len(trainset) == 2
    assert trainset[0].problem == "2+2?"
    assert trainset[0].answer == 4.0
