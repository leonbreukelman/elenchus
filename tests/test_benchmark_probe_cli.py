"""Tests for benchmark_probe CLI helper paths."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_probe import ProbeBenchmarkSummary, _load_baseline_summary, _load_selected_problems, _print_comparison


def test_load_selected_problems_uses_preset(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake_loader(*, dataset: str, split: str, dataset_path=None, limit=0):
        calls.append((dataset, split, dataset_path))
        return [
            {
                "question": f"q-{dataset}",
                "expected_answer": 1.0,
                "category": "arithmetic",
                "source": dataset,
                "source_id": f"{dataset}-{split}-1",
            }
        ]

    monkeypatch.setattr("benchmark_probe.load_calibration_problems", fake_loader)

    problems, source = _load_selected_problems(
        dataset="builtin",
        split="train",
        dataset_path=None,
        preset="official-core",
    )

    assert [c[:2] for c in calls] == [("gsm8k", "train"), ("math", "train")]
    assert len(problems) == 2
    assert source == "preset=official-core split=train"


def test_load_baseline_summary_reads_summary(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"summary": {"answer_accuracy_pct": 55.0}}))

    summary = _load_baseline_summary(baseline)

    assert summary["answer_accuracy_pct"] == 55.0


def test_print_comparison_returns_deltas():
    summary = ProbeBenchmarkSummary(
        model_fast="fast",
        model_capable="capable",
        total=10,
        answer_correct=7,
        answer_accuracy_pct=70.0,
        verified=5,
        uncertain=2,
        refuted=2,
        probe_errors=1,
        correct_and_verified=4,
        correct_and_refuted=1,
        wrong_and_verified=1,
        wrong_and_refuted=1,
        mean_probe_score=0.65,
        mean_mechanism_score=0.60,
        mean_latency_s=4.5,
        total_elapsed_s=45.0,
    )

    baseline = {
        "answer_accuracy_pct": 60.0,
        "verified": 4,
        "refuted": 3,
        "wrong_and_verified": 2,
        "correct_and_refuted": 2,
        "mean_probe_score": 0.55,
        "mean_mechanism_score": 0.50,
        "mean_latency_s": 5.5,
    }

    deltas = _print_comparison(summary, baseline)

    assert deltas["answer_accuracy_pct"] == 10.0
    assert deltas["verified"] == 1.0
    assert deltas["mean_latency_s"] == -1.0
