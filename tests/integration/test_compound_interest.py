"""End-to-end integration test with the compound interest example.

Requires ANTHROPIC_API_KEY environment variable.
Run with: uv run pytest tests/integration/ -v -s
"""

import os

import pytest

from elenchus.engine import build_engine_graph

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.mark.asyncio
async def test_compound_interest_full_pipeline():
    """The spec's walkthrough: $10,000 at 5% compounded monthly for 3 years."""
    graph = build_engine_graph()
    result = await graph.ainvoke(
        {"problem": "$10,000 invested at 5% annual interest compounded monthly. What is the value after 3 years?"}
    )

    verified = result["verified_result"]

    assert verified.answer is not None
    answer = float(verified.answer)
    assert 11600 < answer < 11650, f"Expected ~11614.72, got {answer}"
    # Confidence depends on council agreement — unanimous gives ~1.0, majority ~0.5
    assert verified.confidence > 0.4

    print(f"\nAnswer: {answer}")
    print(f"Confidence: {verified.confidence}")
    print(f"Probe verdict: {verified.probe_verdict}")
    if verified.sensitivity_map:
        print(f"Sensitivity map: {verified.sensitivity_map}")


@pytest.mark.asyncio
async def test_simple_equation():
    """Simple equation: should solve correctly, may skip probe."""
    graph = build_engine_graph()
    result = await graph.ainvoke({"problem": "Solve for x: 2x + 3 = 7"})

    verified = result["verified_result"]
    answer = float(verified.answer)
    assert abs(answer - 2.0) < 0.01, f"Expected 2.0, got {answer}"
