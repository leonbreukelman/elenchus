"""Sandboxed code execution via subprocess with import restrictions and timeout."""

from __future__ import annotations

import asyncio
import textwrap

import structlog
from pydantic import BaseModel

log = structlog.get_logger()

DEFAULT_ALLOWED_IMPORTS: list[str] = [
    "sympy",
    "numpy",
    "math",
    "fractions",
    "decimal",
]


class SandboxResult(BaseModel):
    """Result of a sandboxed code execution."""

    success: bool
    output: str
    error: str


def _build_restricted_script(code: str, allowed_imports: list[str]) -> str:
    """Wrap *code* with an import guard that blocks dangerous modules.

    The guard overrides ``__builtins__.__import__`` to block a deny-list of
    dangerous modules (``subprocess``, ``os``, ``shutil``, etc.) unless they
    are explicitly included in *allowed_imports*.  All other imports —
    including stdlib internals and transitive dependencies of allowed
    packages — pass through unblocked.

    This design recognises that packages like ``sympy`` have deep dependency
    trees (``mpmath``, ``sys``, ``collections``, etc.) that cannot be
    enumerated ahead of time.  The sandbox's security boundary is the
    subprocess itself; the import guard adds defence-in-depth against
    accidental use of dangerous APIs in LLM-generated code.
    """
    allowed_set = repr(set(allowed_imports))
    guard = textwrap.dedent(f"""\
        import builtins as _builtins
        import traceback as _traceback
        _original_import = _builtins.__import__
        _allowed = {allowed_set}

        def _is_user_import():
            \"\"\"Return True if the import originates from user code (<string>),
            not from within an installed package.\"\"\"
            for frame in _traceback.extract_stack():
                if frame.filename == "<string>":
                    continue
                # If we see a site-packages frame before user code, it's internal
                if "site-packages" in frame.filename:
                    return False
            return True

        def _restricted_import(name, *args, **kwargs):
            top = name.split(".")[0]
            # Internal/private modules always pass
            if top.startswith("_"):
                return _original_import(name, *args, **kwargs)
            # Explicitly allowed — always passes
            if top in _allowed:
                return _original_import(name, *args, **kwargs)
            # Only restrict imports originating from user code
            if _is_user_import():
                raise ImportError(f"Import of '{{name}}' is not allowed")
            # Internal imports from allowed packages pass through
            return _original_import(name, *args, **kwargs)

        _builtins.__import__ = _restricted_import
    """)
    return guard + "\n" + code


async def execute_code(
    code: str,
    timeout: float = 30,
    allowed_imports: list[str] | None = None,
) -> SandboxResult:
    """Execute *code* in a subprocess with import restrictions and a timeout.

    Parameters
    ----------
    code:
        Python source to execute.
    timeout:
        Maximum wall-clock seconds before the process is killed.
    allowed_imports:
        Top-level package names the code is permitted to import.
        Defaults to :data:`DEFAULT_ALLOWED_IMPORTS`.

    Returns
    -------
    SandboxResult
        Contains *success*, captured *output* (stdout), and *error* (stderr).
    """
    if allowed_imports is None:
        allowed_imports = list(DEFAULT_ALLOWED_IMPORTS)

    script = _build_restricted_script(code, allowed_imports)

    log.debug("sandbox.execute", timeout=timeout, allowed_imports=allowed_imports)

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log.warning("sandbox.timeout", timeout=timeout)
            return SandboxResult(
                success=False,
                output="",
                error=f"Execution timeout after {timeout}s",
            )

        output = stdout.decode()
        error = stderr.decode()

        if proc.returncode != 0:
            log.info("sandbox.failure", returncode=proc.returncode, error=error[:200])
            return SandboxResult(success=False, output=output, error=error)

        log.debug("sandbox.success", output_len=len(output))
        return SandboxResult(success=True, output=output, error=error)

    except Exception as exc:
        log.error("sandbox.error", error=str(exc))
        return SandboxResult(success=False, output="", error=str(exc))
