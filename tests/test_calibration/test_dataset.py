"""Tests for the calibration dataset."""

import json

import pytest

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


def test_unknown_dataset_raises_value_error():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_calibration_problems(dataset="does-not-exist")


def test_load_from_local_json_file(tmp_path):
    dataset_file = tmp_path / "calibration.json"
    rows = [
        {
            "question": "What is 2 + 2?",
            "expected_answer": 4,
            "category": "arithmetic",
            "source": "local",
            "source_id": "local-1",
        }
    ]
    dataset_file.write_text(json.dumps(rows))

    problems = load_calibration_problems(dataset_path=dataset_file)

    assert len(problems) == 1
    assert problems[0]["question"] == "What is 2 + 2?"
    assert problems[0]["expected_answer"] == 4.0
    assert problems[0]["source_id"] == "local-1"


def test_load_gsm8k_with_monkeypatched_load_dataset(monkeypatch):
    def fake_load_dataset(name, subset=None, split=None):
        assert name == "openai/gsm8k"
        assert subset == "main"
        assert split == "train"
        return [
            {
                "question": "A car goes 60 miles per hour for 2 hours. How far?",
                "answer": "Work... #### 120",
            },
            {
                "question": "This row has no marked answer",
                "answer": "No numeric marker here",
            },
        ]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    problems = load_calibration_problems(dataset="gsm8k", split="train")

    assert len(problems) == 1
    assert problems[0]["expected_answer"] == 120.0
    assert problems[0]["category"] == "rate"
    assert problems[0]["source"] == "gsm8k"
    assert problems[0]["source_id"] == "gsm8k-train-0"
