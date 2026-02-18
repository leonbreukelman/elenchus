"""Consensus engine — aggregates councilor results with tolerance-aware matching."""

from __future__ import annotations

from elenchus import parse_number
from elenchus.state import ConsensusResult, CouncilorResult
from elenchus.tools.sympy_tools import answers_match_numeric


def evaluate_consensus(
    results: list[CouncilorResult],
    rel_tol: float = 1e-3,
) -> ConsensusResult:
    """Evaluate consensus across councilor results.

    Groups answers by numeric equivalence (within *rel_tol*), then classifies
    the outcome as ``"unanimous"``, ``"majority"``, or ``"none"``.

    Parameters
    ----------
    results:
        List of councilor results to evaluate.
    rel_tol:
        Relative tolerance for numeric answer comparison.

    Returns
    -------
    ConsensusResult
        The consensus answer, agreement level, confidence, and dissenters.
    """
    # Filter out results where answer is None
    valid = [r for r in results if r.answer is not None]

    if not valid:
        return ConsensusResult(
            answer=None,
            agreement="none",
            confidence=0.0,
            dissenting_strategies=[],
        )

    # Group by answer equivalence
    groups: list[list[CouncilorResult]] = []
    for result in valid:
        placed = False
        for group in groups:
            representative = group[0].answer
            try:
                if answers_match_numeric(parse_number(result.answer), parse_number(representative), rel_tol):
                    group.append(result)
                    placed = True
                    break
            except (ValueError, TypeError):
                if result.answer == representative:
                    group.append(result)
                    placed = True
                    break
        if not placed:
            groups.append([result])

    # Find largest group
    groups.sort(key=len, reverse=True)
    largest = groups[0]
    total = len(valid)

    if len(largest) == total:
        # Unanimous
        avg_confidence = sum(r.confidence for r in valid) / total
        return ConsensusResult(
            answer=largest[0].answer,
            agreement="unanimous",
            confidence=avg_confidence,
            dissenting_strategies=[],
        )

    if len(largest) >= 2:
        # Majority
        majority_strategies = {r.strategy for r in largest}
        dissenters = [r.strategy for r in valid if r.strategy not in majority_strategies]
        avg_confidence = sum(r.confidence for r in largest) / len(largest)
        return ConsensusResult(
            answer=largest[0].answer,
            agreement="majority",
            confidence=avg_confidence,
            dissenting_strategies=dissenters,
        )

    # No agreement — pick highest-confidence answer, halve average confidence
    best = max(valid, key=lambda r: r.confidence)
    avg_confidence = sum(r.confidence for r in valid) / total
    return ConsensusResult(
        answer=best.answer,
        agreement="none",
        confidence=avg_confidence / 2,
        dissenting_strategies=[],
    )
