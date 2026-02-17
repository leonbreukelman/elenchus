# Probe Hardening: Type-Safe Perturbations + Judge Calibration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Deutsch Probe dtype-aware (integers stay integers, probabilities stay bounded) and improve the mechanism judge by reframing it as delta verification instead of open-ended evaluation.

**Architecture:** Two independent improvements to existing probe pipeline. Perturbation dtype-awareness threads through the Constraint model, extractor prompt, and perturbation generator. Judge calibration is an isolated prompt change in mechanism_judge.py. No new dependencies, no architectural changes.

**Tech Stack:** Python, Pydantic, pytest, Anthropic API (mocked in tests)

---

### Task 1: Add dtype enum to Constraint model

**Files:**
- Modify: `src/elenchus/state.py:61-68`
- Test: `tests/test_state.py`

**Step 1: Write the failing test**

Add to `tests/test_state.py`:

```python
from elenchus.state import Constraint, ConstraintDtype


def test_constraint_dtype_defaults_to_continuous():
    c = Constraint(
        name="rate",
        original_value=0.05,
        dtype="continuous",
        role="interest rate",
        perturbation_range=(0.01, 0.20),
    )
    assert c.dtype == ConstraintDtype.CONTINUOUS


def test_constraint_dtype_integer():
    c = Constraint(
        name="people",
        original_value=5,
        dtype="integer",
        role="number of people",
        perturbation_range=(1, 20),
    )
    assert c.dtype == ConstraintDtype.INTEGER


def test_constraint_dtype_probability():
    c = Constraint(
        name="chance",
        original_value=0.3,
        dtype="probability",
        role="probability of rain",
        perturbation_range=(0.0, 1.0),
    )
    assert c.dtype == ConstraintDtype.PROBABILITY


def test_constraint_backward_compat_numeric_maps_to_continuous():
    """Old 'numeric' dtype should still work, mapping to continuous."""
    c = Constraint(
        name="x",
        original_value=10,
        dtype="numeric",
        role="some value",
        perturbation_range=(1, 100),
    )
    assert c.dtype == ConstraintDtype.CONTINUOUS
```

**Step 2: Run test to verify it fails**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_state.py -v -k "dtype"`
Expected: FAIL — `ConstraintDtype` does not exist yet

**Step 3: Write minimal implementation**

In `src/elenchus/state.py`, add the enum and update `Constraint`:

```python
class ConstraintDtype(str, Enum):
    """Type classification for perturbation constraints."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    PROBABILITY = "probability"
```

Update `Constraint.dtype` field:

```python
class Constraint(BaseModel):
    """A named constraint extracted from the problem for perturbation testing."""

    name: str
    original_value: Any
    dtype: ConstraintDtype
    role: str
    perturbation_range: tuple[float, float]

    @field_validator("dtype", mode="before")
    @classmethod
    def _coerce_legacy_numeric(cls, v: str) -> str:
        if v == "numeric":
            return "continuous"
        return v
```

Requires adding `field_validator` to the pydantic import.

**Step 4: Run test to verify it passes**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_state.py -v -k "dtype"`
Expected: PASS

**Step 5: Run full test suite to check backward compat**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/ -v --tb=short`
Expected: All existing tests PASS (they all use `dtype="numeric"` which maps to `continuous`)

**Step 6: Commit**

```bash
git add src/elenchus/state.py tests/test_state.py
git commit -m "feat(probe): add ConstraintDtype enum with backward-compat numeric mapping"
```

---

### Task 2: Make perturbation generator dtype-aware

**Files:**
- Modify: `src/elenchus/probe/perturbation.py`
- Modify: `tests/test_probe/test_perturbation.py`

**Step 1: Write the failing tests**

Add to `tests/test_probe/test_perturbation.py`:

```python
from elenchus.state import ConstraintDtype


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
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_probe/test_perturbation.py -v -k "integer or probability or continuous_perturbation"`
Expected: FAIL — integer test gets float values like 8.4

**Step 3: Write minimal implementation**

In `src/elenchus/probe/perturbation.py`, add a post-processing function and apply it:

```python
from elenchus.state import Constraint, ConstraintDtype, Perturbation


def _coerce_to_dtype(value: float, constraint: Constraint) -> float | int:
    """Round/clamp the perturbation value to match the constraint's dtype."""
    if constraint.dtype == ConstraintDtype.INTEGER:
        return int(round(value))
    if constraint.dtype == ConstraintDtype.PROBABILITY:
        return max(0.0, min(1.0, value))
    return value
```

Then wrap each shift function's return in every call site. The cleanest approach: apply `_coerce_to_dtype` inside `generate_perturbations` after each value is computed, replacing the three `Perturbation(... new_value=...)` blocks:

```python
def generate_perturbations(
    constraints: list[Constraint],
    budget: int = 3,
) -> list[Perturbation]:
    """Generate up to `budget` perturbations across constraints."""
    if not constraints:
        return []

    effective_budget = min(budget, len(constraints))
    perturbations: list[Perturbation] = []

    if effective_budget >= 1:
        c = constraints[0]
        perturbations.append(
            Perturbation(
                constraint=c,
                new_value=_coerce_to_dtype(_moderate_shift(c), c),
                rationale="Primary sensitivity test — moderate shift on central constraint",
            )
        )

    if effective_budget >= 2:
        c = constraints[1 % len(constraints)]
        perturbations.append(
            Perturbation(
                constraint=c,
                new_value=_coerce_to_dtype(_boundary_value(c), c),
                rationale="Boundary behavior test — errors amplify at extremes",
            )
        )

    if effective_budget >= 3:
        c = constraints[2 % len(constraints)]
        perturbations.append(
            Perturbation(
                constraint=c,
                new_value=_coerce_to_dtype(_subtle_shift(c), c),
                rationale="Quantitative precision test — small shift tests magnitude accuracy",
            )
        )

    logger.info("perturbations_generated", count=len(perturbations))
    return perturbations
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_probe/test_perturbation.py -v`
Expected: All PASS (new and existing)

**Step 5: Commit**

```bash
git add src/elenchus/probe/perturbation.py tests/test_probe/test_perturbation.py
git commit -m "feat(probe): dtype-aware perturbation — integers round, probabilities clamp to [0,1]"
```

---

### Task 3: Update extractor prompt to request dtype

**Files:**
- Modify: `src/elenchus/probe/extractor.py`
- Modify: `tests/test_probe/test_extractor.py`

**Step 1: Write the failing test**

Add to `tests/test_probe/test_extractor.py`:

```python
@pytest.mark.asyncio
async def test_extract_constraints_prompt_requests_dtype():
    """The extraction prompt should ask for dtype classification."""
    mock_response = AsyncMock()
    mock_response.content = [
        AsyncMock(
            text='[{"name":"people","original_value":5,"dtype":"integer",'
            '"role":"number of workers","perturbation_range":[1,20]}]'
        )
    ]

    with patch("elenchus.probe.extractor._get_client") as mock_client:
        mock_create = AsyncMock(return_value=mock_response)
        mock_client.return_value.messages.create = mock_create
        constraints = await extract_constraints(
            problem="5 workers build a wall in 10 days",
            solution="10 days",
        )

    assert len(constraints) == 1
    assert constraints[0].dtype.value == "integer"

    # Verify the prompt includes dtype guidance
    call_args = mock_create.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    assert "integer" in prompt_content
    assert "probability" in prompt_content
```

**Step 2: Run test to verify it fails**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_probe/test_extractor.py::test_extract_constraints_prompt_requests_dtype -v`
Expected: FAIL — prompt doesn't mention "integer" or "probability"

**Step 3: Update the extraction prompts**

In `src/elenchus/probe/extractor.py`, update both `EXTRACTION_PROMPT` and `EXTRACTION_WITH_CODE_PROMPT`. Change the dtype field description from:

```
- dtype: "numeric" (for now, always numeric)
```

to:

```
- dtype: one of "continuous" (real-valued like prices, rates, distances), \
"integer" (whole numbers like counts of people, items, days), or \
"probability" (values bounded to [0, 1])
```

Apply this change in both prompts. No other code changes needed — the `Constraint` model's validator handles the rest.

**Step 4: Run tests to verify they pass**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_probe/test_extractor.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/elenchus/probe/extractor.py tests/test_probe/test_extractor.py
git commit -m "feat(probe): extractor now requests dtype — integer, continuous, or probability"
```

---

### Task 4: Reframe judge prompt as delta verification

**Files:**
- Modify: `src/elenchus/probe/mechanism_judge.py`
- Modify: `tests/test_probe/test_mechanism_judge.py`

**Step 1: Write the failing test**

Add to `tests/test_probe/test_mechanism_judge.py`:

```python
@pytest.mark.asyncio
async def test_judge_prompt_includes_delta_information():
    """The judge prompt should include the numeric delta for verification."""
    from elenchus.probe.mechanism_judge import judge_mechanism

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text='{"score": 0.9, "reasoning": "Delta is correct."}')
    ]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("elenchus.probe.mechanism_judge._get_client", return_value=mock_client):
        await judge_mechanism(
            constraint_role="annual interest rate",
            original_value=0.05,
            new_value=0.08,
            original_answer=11614.72,
            actual_answer=12682.42,
            predicted_reasoning="Higher rate increases compounding.",
        )

    # Inspect the prompt sent to the model
    call_args = mock_client.messages.create.call_args
    prompt_text = call_args.kwargs["messages"][0]["content"]

    # Must contain the computed delta
    assert "1067.7" in prompt_text or "1067.70" in prompt_text  # 12682.42 - 11614.72
    # Must frame as verification, not open evaluation
    assert "verify" in prompt_text.lower() or "justify" in prompt_text.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_probe/test_mechanism_judge.py::test_judge_prompt_includes_delta_information -v`
Expected: FAIL — current prompt doesn't include delta

**Step 3: Update the judge prompt**

In `src/elenchus/probe/mechanism_judge.py`, replace `JUDGE_PROMPT`:

```python
JUDGE_PROMPT = """\
You are verifying whether a mathematical reasoning explanation justifies \
a specific numeric change.

A constraint was perturbed:
- What changed: {constraint_role}
- Original value: {original_value}
- New value: {new_value}

The answers:
- Original answer: {original_answer}
- Actual answer (ground truth): {actual_answer}
- Delta (actual change): {delta}

The councilor's reasoning about why the answer changed:
"{predicted_reasoning}"

Verify whether the stated reasoning mathematically justifies a delta of \
{delta}. Score 0.0-1.0:
1. Does the reasoning identify the correct mathematical relationship?
2. Is the explained mechanism consistent with a change of {delta}?
3. Is the explanation specific (names the relationship, not generic)?

Return ONLY valid JSON with:
- "score": float between 0.0 and 1.0
- "reasoning": one sentence explaining the score

No markdown fences or extra text.\
"""
```

Update the `judge_mechanism` function to compute and pass the delta:

```python
async def judge_mechanism(
    constraint_role: str,
    original_value: float,
    new_value: float,
    original_answer: float,
    actual_answer: float,
    predicted_reasoning: str,
) -> float:
    if not predicted_reasoning or not predicted_reasoning.strip():
        logger.debug("mechanism_judge_no_reasoning")
        return 0.5

    delta = actual_answer - original_answer

    prompt = JUDGE_PROMPT.format(
        constraint_role=constraint_role,
        original_value=original_value,
        new_value=new_value,
        original_answer=original_answer,
        actual_answer=actual_answer,
        delta=round(delta, 2),
        predicted_reasoning=predicted_reasoning,
    )
    # ... rest unchanged
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/test_probe/test_mechanism_judge.py -v`
Expected: All PASS (new and existing)

**Step 5: Run full test suite**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/elenchus/probe/mechanism_judge.py tests/test_probe/test_mechanism_judge.py
git commit -m "feat(probe): judge prompt reframed as delta verification for better Haiku accuracy"
```

---

### Task 5: Final integration check

**Files:**
- None created or modified

**Step 1: Run full test suite**

Run: `cd /home/leonb/projects/elenchus && uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run linter**

Run: `cd /home/leonb/projects/elenchus && uv run ruff check src/ tests/`
Expected: Clean

**Step 3: Push**

```bash
git push origin main
```
