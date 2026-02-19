"""Tests for elenchus.extract_json and elenchus.parse_number."""

from __future__ import annotations

import json

import pytest

from elenchus import extract_json, parse_number

# ---------------------------------------------------------------------------
# extract_json — existing behaviour
# ---------------------------------------------------------------------------


class TestExtractJsonExisting:
    def test_valid_json_object(self):
        """Direct valid JSON parses on the first attempt."""
        raw = '{"answer": 42, "reasoning": "trivial"}'
        result = extract_json(raw)
        assert result == {"answer": 42, "reasoning": "trivial"}

    def test_valid_json_array(self):
        result = extract_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_markdown_fences(self):
        """JSON wrapped in markdown code fences is extracted."""
        raw = '```json\n{"answer": 3.14}\n```'
        result = extract_json(raw)
        assert result == {"answer": 3.14}

    def test_markdown_fences_no_language(self):
        raw = '```\n{"answer": 7}\n```'
        result = extract_json(raw)
        assert result == {"answer": 7}

    def test_surrounding_prose(self):
        """JSON embedded in prose is found by bracket matching."""
        raw = 'Here is my answer:\n{"answer": 2.5, "reasoning": "half of 5"}\nDone.'
        result = extract_json(raw)
        assert result == {"answer": 2.5, "reasoning": "half of 5"}

    def test_no_json_raises(self):
        """Text with no JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            extract_json("This is just plain text with no JSON.")


# ---------------------------------------------------------------------------
# extract_json — new sanitization behaviour (step 4)
# ---------------------------------------------------------------------------


class TestExtractJsonSanitization:
    def test_literal_newlines_in_string_values(self):
        """Literal newlines inside JSON string values are repaired."""
        raw = '{"answer": 42, "reasoning": "Step 1: compute x\nStep 2: solve for y"}'
        result = extract_json(raw)
        assert result["answer"] == 42
        assert "Step 1" in result["reasoning"]
        assert "Step 2" in result["reasoning"]

    def test_literal_tabs_in_string_values(self):
        """Literal tabs inside JSON string values are repaired."""
        raw = '{"answer": 7, "reasoning": "column1\tcolumn2"}'
        result = extract_json(raw)
        assert result["answer"] == 7
        assert "column1" in result["reasoning"]

    def test_mixed_control_characters(self):
        """Mixed newlines, carriage returns, and tabs inside strings."""
        raw = '{"answer": 1, "reasoning": "line1\r\nline2\ttab"}'
        result = extract_json(raw)
        assert result["answer"] == 1
        assert "line1" in result["reasoning"]
        assert "line2" in result["reasoning"]

    def test_multiline_reasoning_with_prose(self):
        """Real-world pattern: LLM wraps multiline reasoning in JSON with prose around it."""
        raw = (
            "Let me solve this step by step.\n"
            '{"answer": 0.25, "reasoning": "We compute 1/4 = 0.25.\n'
            "Then we verify:\n"
            '0.25 * 4 = 1. Correct.", "confidence": 0.95}\n'
            "I hope that helps!"
        )
        result = extract_json(raw)
        assert result["answer"] == 0.25
        assert result["confidence"] == 0.95

    def test_already_escaped_characters_untouched(self):
        """Already-escaped \\n sequences are not double-escaped."""
        raw = '{"answer": 1, "reasoning": "line1\\nline2"}'
        result = extract_json(raw)
        assert result["answer"] == 1
        assert "line1\nline2" == result["reasoning"]

    def test_newlines_outside_strings_preserved(self):
        """Newlines in JSON structure (outside strings) are fine — valid JSON."""
        raw = '{\n  "answer": 5,\n  "reasoning": "done"\n}'
        result = extract_json(raw)
        assert result["answer"] == 5

    def test_invalid_escape_dollar(self):
        r"""Invalid \$ escape inside string value is repaired."""
        raw = r'{"answer": 1, "reasoning": "costs \$5"}'
        result = extract_json(raw)
        assert result["answer"] == 1
        assert "$5" in result["reasoning"]

    def test_invalid_escape_parens(self):
        r"""Invalid \( and \) escapes inside string values are repaired."""
        raw = r'{"answer": 2, "reasoning": "solve \(x + 1\) = 3"}'
        result = extract_json(raw)
        assert result["answer"] == 2
        assert "(x + 1)" in result["reasoning"]

    def test_invalid_escape_mixed_with_valid(self):
        r"""Valid escapes like \n are preserved while invalid ones like \$ are fixed."""
        raw = '{"answer": 3, "reasoning": "line1\\nline2 costs \\$10"}'
        result = extract_json(raw)
        assert result["answer"] == 3
        assert "line1\nline2" in result["reasoning"]
        assert "$10" in result["reasoning"]

    def test_backslash_outside_string_untouched(self):
        r"""Backslashes outside JSON string values are not altered."""
        raw = '{"answer": 4, "reasoning": "ok"}'
        result = extract_json(raw)
        assert result["answer"] == 4


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------


class TestParseNumber:
    def test_int(self):
        assert parse_number(42) == 42.0

    def test_float(self):
        assert parse_number(3.14) == 3.14

    def test_string_float(self):
        assert parse_number("0.25") == 0.25

    def test_string_int(self):
        assert parse_number("7") == 7.0

    def test_fraction_quarter(self):
        assert parse_number("1/4") == 0.25

    def test_fraction_three_halves(self):
        assert parse_number("3/2") == 1.5

    def test_negative_fraction(self):
        assert parse_number("-3/4") == -0.75

    def test_sympy_rational(self):
        assert parse_number("sympy.Rational(1, 2)") == 0.5

    def test_sympy_rational_negative(self):
        assert parse_number("sympy.Rational(-3, 4)") == -0.75

    def test_string_with_whitespace(self):
        assert parse_number("  42.0  ") == 42.0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_number("not a number")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            parse_number(None)

    def test_list_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            parse_number([1, 2])

    def test_division_by_zero_fraction(self):
        with pytest.raises(ValueError, match="Division by zero"):
            parse_number("1/0")

    def test_division_by_zero_sympy(self):
        with pytest.raises(ValueError, match="Division by zero"):
            parse_number("sympy.Rational(1, 0)")
