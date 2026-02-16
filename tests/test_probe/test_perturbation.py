from elenchus.probe.perturbation import generate_perturbations
from elenchus.state import Constraint, Perturbation


def _make_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="principal",
            original_value=10000,
            dtype="numeric",
            role="initial investment",
            perturbation_range=(1000, 50000),
        ),
        Constraint(
            name="rate",
            original_value=0.05,
            dtype="numeric",
            role="annual interest rate",
            perturbation_range=(0.01, 0.20),
        ),
        Constraint(
            name="time",
            original_value=3,
            dtype="numeric",
            role="years",
            perturbation_range=(1, 30),
        ),
    ]


def test_generates_up_to_budget():
    constraints = _make_constraints()
    perturbations = generate_perturbations(constraints, budget=3)
    assert len(perturbations) == 3
    assert all(isinstance(p, Perturbation) for p in perturbations)


def test_respects_budget():
    constraints = _make_constraints()
    perturbations = generate_perturbations(constraints, budget=2)
    assert len(perturbations) == 2


def test_budget_capped_by_constraint_count():
    constraints = _make_constraints()[:1]
    perturbations = generate_perturbations(constraints, budget=3)
    assert len(perturbations) == 1


def test_moderate_shift_within_range():
    constraints = _make_constraints()
    perturbations = generate_perturbations(constraints, budget=1)
    p = perturbations[0]
    lo, hi = p.constraint.perturbation_range
    assert lo <= p.new_value <= hi


def test_perturbation_values_differ_from_original():
    constraints = _make_constraints()
    perturbations = generate_perturbations(constraints, budget=3)
    for p in perturbations:
        assert p.new_value != p.constraint.original_value


def test_each_perturbation_has_rationale():
    constraints = _make_constraints()
    perturbations = generate_perturbations(constraints, budget=3)
    for p in perturbations:
        assert p.rationale != ""


def test_empty_constraints():
    perturbations = generate_perturbations([], budget=3)
    assert perturbations == []
