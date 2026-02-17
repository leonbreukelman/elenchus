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


# --- dtype-aware perturbation tests ---


def _make_integer_constraint() -> Constraint:
    return Constraint(
        name="people",
        original_value=12,
        dtype="integer",
        role="number of people",
        perturbation_range=(1, 50),
    )


def _make_probability_constraint() -> Constraint:
    return Constraint(
        name="chance",
        original_value=0.3,
        dtype="probability",
        role="probability of event",
        perturbation_range=(0.0, 1.0),
    )


def test_integer_perturbation_produces_whole_numbers():
    """Integer constraints must produce integer-valued perturbations."""
    constraints = [_make_integer_constraint()]
    for _ in range(20):  # Run multiple times since random
        perturbations = generate_perturbations(constraints, budget=1)
        value = perturbations[0].new_value
        assert value == int(value), f"Expected integer, got {value}"
        assert value >= 1
        assert value <= 50


def test_probability_perturbation_stays_in_unit_interval():
    """Probability constraints must stay in [0, 1]."""
    constraints = [_make_probability_constraint()]
    for _ in range(20):
        perturbations = generate_perturbations(constraints, budget=1)
        value = perturbations[0].new_value
        assert 0.0 <= value <= 1.0, f"Probability out of bounds: {value}"


def test_integer_boundary_value_is_whole():
    """Boundary perturbation on integer constraint must also be integer."""
    constraints = [
        _make_integer_constraint(),
        _make_integer_constraint(),
    ]
    for _ in range(20):
        perturbations = generate_perturbations(constraints, budget=2)
        for p in perturbations:
            assert p.new_value == int(p.new_value), f"Expected integer, got {p.new_value}"


def test_continuous_perturbation_unchanged():
    """Continuous constraints still produce float values (existing behavior)."""
    constraints = _make_constraints()  # Uses existing helper, dtype="numeric" -> continuous
    perturbations = generate_perturbations(constraints, budget=1)
    value = perturbations[0].new_value
    assert isinstance(value, float)
