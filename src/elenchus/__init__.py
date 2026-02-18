"""Elenchus — neuro-symbolic math verification engine."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> object:
    """Parse JSON from LLM output, handling fences and surrounding prose.

    Tries in order:
    1. Direct parse (already valid JSON)
    2. Strip markdown code fences and parse
    3. Find first { or [ and parse from there
    """
    stripped = text.strip()

    # Try direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first JSON object or array in the text
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
                    return json.loads(stripped[start : i + 1])

    raise json.JSONDecodeError("No JSON found in text", text, 0)
