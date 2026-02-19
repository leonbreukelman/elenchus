"""Elenchus — neuro-symbolic math verification engine."""

from __future__ import annotations

import json
import re


def _sanitize_json_controls(text: str) -> str:
    """Replace literal control characters inside JSON string values.

    Walks the text character-by-character, tracking whether we are inside a
    JSON string (between unescaped ``"`` delimiters).  Inside a string, literal
    newlines, carriage returns, and tabs are replaced with their JSON-escaped
    equivalents (``\\n``, ``\\r``, ``\\t``).  Characters outside strings and
    already-escaped sequences are left untouched.
    """
    out: list[str] = []
    in_string = False
    escape_next = False

    _VALID_JSON_ESCAPES = set('"\\/bfnrtu')

    for ch in text:
        if escape_next:
            # Inside a string, only valid JSON escape targets are allowed.
            # If the character is NOT a valid target, drop the preceding
            # backslash (already appended) so e.g. \$ becomes just $.
            if in_string and ch not in _VALID_JSON_ESCAPES:
                out.pop()  # remove the backslash
            out.append(ch)
            escape_next = False
            continue

        if ch == "\\":
            out.append(ch)
            if in_string:
                escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue

        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue

        out.append(ch)

    return "".join(out)


def extract_json(text: str) -> object:
    """Parse JSON from LLM output, handling fences and surrounding prose.

    Tries in order:
    1. Direct parse (already valid JSON)
    2. Strip markdown code fences and parse
    3. Find first { or [ and parse from there
    4. Sanitize control characters inside strings, then re-parse
    """
    stripped = text.strip()

    # Step 1 — direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Step 2 — strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3 — find first JSON object or array in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        if start == -1:
            continue
        # Find the matching closing bracket by counting nesting
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(stripped)):
            c = stripped[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Fall through to step 4
                        break

    # Step 4 — sanitize control characters inside JSON strings, then re-parse
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stripped.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(stripped)):
            c = stripped[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    sanitized = _sanitize_json_controls(candidate)
                    return json.loads(sanitized)

    raise json.JSONDecodeError("No JSON found in text", text, 0)


_FRACTION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$")
_SYMPY_RATIONAL_RE = re.compile(
    r"^sympy\.Rational\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$"
)


def parse_number(value: object) -> float:
    """Convert a value to float, handling fractions, SymPy expressions, and strings.

    Handles: 0.25, "0.25", "1/4", "3/2", "sympy.Rational(1, 2)", integer strings.
    Raises ``ValueError`` if conversion fails.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"Cannot convert {type(value).__name__} to number: {value!r}")

    s = value.strip()

    # Try plain float
    try:
        return float(s)
    except ValueError:
        pass

    # Try fraction pattern  a/b
    m = _FRACTION_RE.match(s)
    if m:
        numerator = float(m.group(1))
        denominator = float(m.group(2))
        if denominator == 0:
            raise ValueError(f"Division by zero in fraction: {s!r}")
        return numerator / denominator

    # Try sympy.Rational(a, b) pattern
    m = _SYMPY_RATIONAL_RE.match(s)
    if m:
        numerator = float(m.group(1))
        denominator = float(m.group(2))
        if denominator == 0:
            raise ValueError(f"Division by zero in sympy.Rational: {s!r}")
        return numerator / denominator

    raise ValueError(f"Cannot parse as number: {value!r}")
