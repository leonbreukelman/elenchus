"""Phase 1: Extract perturbable constraints from a problem."""

from __future__ import annotations

import structlog

from elenchus import extract_json
from elenchus.config import get_model_config
from elenchus.llm import complete
from elenchus.state import Constraint

logger = structlog.get_logger()

EXTRACTION_PROMPT = """\
Given this math problem and its solution, identify every input constraint that the answer depends on.

For each, return a JSON array of objects with:
- name: variable name (snake_case)
- original_value: the numeric value
- dtype: one of "continuous" (real-valued like prices, rates, distances), \
"integer" (whole numbers like counts of people, items, days), or \
"probability" (values bounded to [0, 1])
- role: plain English description of what this constraint represents
- perturbation_range: [min, max] of valid alternative values

Only include constraints where changing the value SHOULD change the answer.
Exclude constants of nature and fixed definitions.

Return ONLY a valid JSON array.

Problem: {problem}
Solution: {solution}"""

EXTRACTION_WITH_CODE_PROMPT = """\
Given this math problem, its solution, and the SymPy code that solved it, \
identify every perturbable input constraint.

IMPORTANT: Use the exact variable names from the code (e.g., "P" not \
"principal_amount"). The constraint name MUST match the variable assignment \
in the code.

For each, return a JSON array of objects with:
- name: the exact variable name from the code
- original_value: the numeric value
- dtype: one of "continuous" (real-valued like prices, rates, distances), \
"integer" (whole numbers like counts of people, items, days), or \
"probability" (values bounded to [0, 1])
- role: plain English description of what this constraint represents
- perturbation_range: [min, max] of valid alternative values

Only include input variables that are directly assigned a literal value in the \
code (e.g., P = 10000). Do NOT include computed variables (e.g., \
A = P*(1+r/n)**(n*t)).

Return ONLY a valid JSON array.

Problem: {problem}
Solution: {solution}
Code:
{symbolic_code}"""


async def extract_constraints(problem: str, solution: str, symbolic_code: str | None = None) -> list[Constraint]:
    """Extract perturbable constraints from a problem-solution pair.

    When *symbolic_code* is provided, the LLM is instructed to use the code's
    variable names as constraint names so they match at substitution time.
    """
    if symbolic_code is not None:
        prompt = EXTRACTION_WITH_CODE_PROMPT.format(problem=problem, solution=solution, symbolic_code=symbolic_code)
    else:
        prompt = EXTRACTION_PROMPT.format(problem=problem, solution=solution)

    model = get_model_config().fast

    response = await complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=get_model_config().max_tokens_fast,
    )

    logger.info("constraint_extraction", raw=response.text)
    parsed = extract_json(response.text)
    return [Constraint(**c) for c in parsed]
