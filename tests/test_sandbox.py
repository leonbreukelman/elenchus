"""Tests for elenchus.tools.sandbox — sandboxed subprocess code execution."""

from __future__ import annotations

from elenchus.tools.sandbox import SandboxResult, execute_code

# ---------------------------------------------------------------------------
# SandboxResult model
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_construction(self):
        r = SandboxResult(success=True, output="4\n", error="")
        assert r.success is True
        assert r.output == "4\n"
        assert r.error == ""


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------


class TestExecuteCode:
    async def test_execute_simple_math(self):
        result = await execute_code("print(2 + 2)")
        assert result.success is True
        assert "4" in result.output

    async def test_execute_sympy_solve(self):
        code = "from sympy import symbols, solve\nx = symbols('x')\nprint(solve(x - 5, x))\n"
        result = await execute_code(code)
        assert result.success is True
        assert "5" in result.output

    async def test_execute_timeout(self):
        code = "import time\ntime.sleep(30)\n"
        result = await execute_code(code, timeout=2, allowed_imports=["time"])
        assert result.success is False
        assert "timeout" in result.error.lower()

    async def test_execute_syntax_error(self):
        result = await execute_code("def foo(:")
        assert result.success is False
        assert result.error != ""

    async def test_execute_disallowed_import(self):
        result = await execute_code("import subprocess", allowed_imports=["sympy"])
        assert result.success is False
        assert result.error != ""

    async def test_execute_returns_last_expression(self):
        code = "from sympy import symbols, solve\nx = symbols('x')\nprint(solve(x**2 - 4, x))\n"
        result = await execute_code(code)
        assert result.success is True
        assert "-2" in result.output
        assert "2" in result.output
