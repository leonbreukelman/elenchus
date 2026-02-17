"""Tests for the calibration dataset."""

from elenchus.calibration.dataset import load_calibration_problems


def test_load_calibration_problems_returns_list():
    """Should return a non-empty list of problem dicts."""
    problems = load_calibration_problems()
    assert isinstance(problems, list)
    assert len(problems) >= 20


def test_each_problem_has_required_fields():
    """Each problem must have question, expected_answer, and category."""
    problems = load_calibration_problems()
    for p in problems:
        assert "question" in p, f"Missing 'question' in {p}"
        assert "expected_answer" in p, f"Missing 'expected_answer' in {p}"
        assert "category" in p, f"Missing 'category' in {p}"
        assert isinstance(p["expected_answer"], (int, float))


def test_problems_cover_multiple_categories():
    """Dataset should cover at least 3 different problem categories."""
    problems = load_calibration_problems()
    categories = {p["category"] for p in problems}
    assert len(categories) >= 3
