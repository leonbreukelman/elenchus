"""Tests for benchmark display helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_display import ActiveProblem, make_active_panel, make_header, make_results_table, row_style
from benchmark_probe import ProbeProblemResult


def _ok_result(**overrides) -> ProbeProblemResult:
    """Factory for a completed problem result."""
    defaults = dict(
        question="A train travels 60 mph for 3 hours...",
        category="rate",
        source="gsm8k",
        expected=180.0,
        consensus_answer=180.0,
        agreement="unanimous",
        answer_correct=True,
        probe_verdict="hard_to_vary",
        probe_score=0.85,
        sensitivity_map={"speed": 0.9},
        mechanism_scores=[0.8, 0.7],
        num_perturbations=3,
        num_ground_truths_ok=2,
        elapsed_s=12.5,
        error=None,
    )
    defaults.update(overrides)
    return ProbeProblemResult(**defaults)


class TestRowStyle:
    def test_correct_hard_to_vary_is_green(self):
        assert row_style(_ok_result()) == "green"

    def test_error_is_red(self):
        r = _ok_result(error="boom", probe_verdict="error", answer_correct=False)
        assert row_style(r) == "red"

    def test_wrong_answer_is_yellow(self):
        r = _ok_result(answer_correct=False, probe_verdict="hard_to_vary")
        assert row_style(r) == "yellow"

    def test_easy_to_vary_is_dim(self):
        r = _ok_result(probe_verdict="easy_to_vary")
        assert row_style(r) == "dim"

    def test_partially_coupled_is_default(self):
        r = _ok_result(probe_verdict="partially_coupled")
        assert row_style(r) == ""


class TestMakeHeader:
    def test_shows_progress(self):
        panel = make_header(
            model_fast="gemini-flash",
            model_capable="gemini-pro",
            completed=5,
            total=20,
            correct=3,
            errors=1,
            elapsed=45.2,
        )
        text = panel.renderable.plain
        assert "5/20" in text
        assert "3 (60.0%)" in text

    def test_zero_completed(self):
        panel = make_header(
            model_fast="fast",
            model_capable="capable",
            completed=0,
            total=10,
            correct=0,
            errors=0,
            elapsed=0.0,
        )
        text = panel.renderable.plain
        assert "0/10" in text


class TestMakeResultsTable:
    def test_empty_table_has_headers(self):
        table = make_results_table([])
        assert table.row_count == 0

    def test_one_result_adds_row(self):
        table = make_results_table([_ok_result()])
        assert table.row_count == 1

    def test_error_result_shows_err(self):
        table = make_results_table([_ok_result(error="fail", probe_verdict="error")])
        assert table.row_count == 1


class TestMakeActivePanel:
    def test_no_active_returns_panel(self):
        panel = make_active_panel([])
        assert panel is not None

    def test_shows_running_problems(self):
        active = [
            ActiveProblem(index=1, question="What is 2+2?", start_time=0.0),
            ActiveProblem(index=3, question="A train goes...", start_time=0.0),
        ]
        panel = make_active_panel(active, now=10.0)
        text = panel.renderable.plain
        assert "#1" in text
        assert "#3" in text
