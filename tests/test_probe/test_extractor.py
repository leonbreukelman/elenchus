from unittest.mock import AsyncMock, patch

import pytest

from elenchus.probe.extractor import extract_constraints
from elenchus.state import Constraint


@pytest.mark.asyncio
async def test_extract_constraints_compound_interest():
    mock_response = AsyncMock()
    mock_response.content = [
        AsyncMock(
            text='[{"name":"principal","original_value":10000,"dtype":"numeric",'
            '"role":"initial investment amount","perturbation_range":[1000,50000]},'
            '{"name":"rate","original_value":0.05,"dtype":"numeric",'
            '"role":"annual interest rate","perturbation_range":[0.01,0.20]},'
            '{"name":"time","original_value":3,"dtype":"numeric",'
            '"role":"investment period in years","perturbation_range":[1,30]}]'
        )
    ]

    with patch("elenchus.probe.extractor._get_client") as mock_client:
        mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
        constraints = await extract_constraints(
            problem="$10,000 at 5% compounded monthly for 3 years",
            solution="$11,614.72",
        )

    assert len(constraints) == 3
    assert all(isinstance(c, Constraint) for c in constraints)
    assert constraints[0].name == "principal"
    assert constraints[1].perturbation_range == (0.01, 0.20)
