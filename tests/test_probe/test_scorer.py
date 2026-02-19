import pytest

from elenchus.probe.scorer import (
    ProbeVerdict,
    compute_direction_score,
    compute_overall_verdict,
    compute_quantitative_score,
)


def test_quantitative_exact_match():
    assert compute_quantitative_score(12702.37, 12702.37) == 1.0


def test_quantitative_close_match():
    score = compute_quantitative_score(12700.0, 12702.37)
    assert score > 0.99


def test_quantitative_wrong():
    score = compute_quantitative_score(25000.0, 12702.37)
    assert score < 0.5


def test_direction_correct_increase():
    assert (
        compute_direction_score(
            instructed=12700.0,
            actual=12702.37,
            original=11614.72,
        )
        == 1.0
    )


def test_direction_correct_decrease():
    assert (
        compute_direction_score(
            instructed=10000.0,
            actual=9500.0,
            original=11614.72,
        )
        == 1.0
    )


def test_direction_wrong():
    assert (
        compute_direction_score(
            instructed=13000.0,
            actual=9500.0,
            original=11614.72,
        )
        == 0.0
    )


def test_verdict_hard_to_vary():
    verdict, recommendation = compute_overall_verdict(0.90)
    assert verdict == ProbeVerdict.HARD_TO_VARY
    assert recommendation == "accept"


def test_verdict_partially_coupled():
    verdict, recommendation = compute_overall_verdict(0.65)
    assert verdict == ProbeVerdict.PARTIALLY_COUPLED
    assert recommendation == "flag_for_review"


def test_verdict_easy_to_vary():
    verdict, recommendation = compute_overall_verdict(0.30)
    assert verdict == ProbeVerdict.EASY_TO_VARY
    assert recommendation == "reject"


def test_verdict_at_boundaries():
    v1, _ = compute_overall_verdict(0.80)
    assert v1 == ProbeVerdict.HARD_TO_VARY

    v2, _ = compute_overall_verdict(0.50)
    assert v2 == ProbeVerdict.PARTIALLY_COUPLED

    v3, _ = compute_overall_verdict(0.49)
    assert v3 == ProbeVerdict.EASY_TO_VARY


def test_alignment_score_uses_provided_mechanism_score():
    """When a real mechanism score is provided, it should replace the default."""
    from elenchus.probe.scorer import compute_alignment_score

    score_with_high_mechanism = compute_alignment_score(
        instructed=100.0,
        actual=100.0,
        original=90.0,
        mechanism_score=0.9,
    )
    score_with_low_mechanism = compute_alignment_score(
        instructed=100.0,
        actual=100.0,
        original=90.0,
        mechanism_score=0.1,
    )
    # quant=1.0, direction=1.0 → 0.4*1 + 0.3*1 + 0.3*mechanism
    assert score_with_high_mechanism == pytest.approx(0.4 + 0.3 + 0.3 * 0.9)
    assert score_with_low_mechanism == pytest.approx(0.4 + 0.3 + 0.3 * 0.1)
    assert score_with_high_mechanism > score_with_low_mechanism
