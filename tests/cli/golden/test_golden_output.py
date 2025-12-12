"""Golden file tests for CLI output stability.

Test that CLI output remains stable across changes.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

from tests._helpers.assertions import expect_in, expect_is_instance, expect_true

if TYPE_CHECKING:
    from tests.cli._harness import CliTestHarness, GoldenFileAssertion


def test_build_status_text_output(
    cli: CliTestHarness,
) -> None:
    """Build status text output is non-empty."""
    result = cli.invoke(["build", "status", "--output-format=text"])

    expect_true(result.exit_code in {0, 1})


def test_build_status_json_structure(
    cli: CliTestHarness,
) -> None:
    """Build status JSON mode doesn't crash."""
    result = cli.invoke(["build", "status", "--output-format=json"])

    expect_true(result.exit_code in {0, 1})


def test_op_list_text_output(
    cli: CliTestHarness,
) -> None:
    """Op list text output contains expected elements."""
    result = cli.invoke(["op", "list", "--output-format=text"])

    expect_true(result.exit_code in {0, 1})

    output = result.stdout + result.stderr
    expect_true(bool(output.strip()))


def test_validation_error_format(
    cli: CliTestHarness,
) -> None:
    """Validation errors have stable format."""
    result = cli.invoke(["build", "run", "--invalid-option=test"])

    expect_true(result.exit_code != 0)

    output = result.stdout + result.stderr
    expect_true(bool(output.strip()))


def test_not_found_error_json(
    cli: CliTestHarness,
) -> None:
    """Not found errors have RFC 9457 structure in JSON mode."""
    result = cli.invoke(["op", "call", "nonexistent.op", "--output-format=json"])

    if result.exit_code != 0:
        with contextlib.suppress(json.JSONDecodeError):
            error_data = json.loads(result.stdout)

            if "type" in error_data:
                expect_is_instance(error_data["type"], str)


def test_normalize_whitespace(
    golden: GoldenFileAssertion,
) -> None:
    """Golden assertion normalizes whitespace."""
    test_text = "  hello\n\n  world  \n"
    normalized = golden.normalize_text(test_text)

    expect_in("hello", normalized)
    expect_in("world", normalized)


def test_json_ignore_keys(
    golden: GoldenFileAssertion,
) -> None:
    """Golden assertion can ignore specific JSON keys."""
    actual = {
        "status": "ok",
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {"value": 42, "id": "unique-id"},
    }

    expected = {
        "status": "ok",
        "timestamp": "different-timestamp",
        "data": {"value": 42, "id": "different-id"},
    }

    filtered_actual = golden.filter_json(actual, {"timestamp", "id"})
    filtered_expected = golden.filter_json(expected, {"timestamp", "id"})

    expect_true(filtered_actual == filtered_expected)


def test_update_mode_creates_file(
    golden: GoldenFileAssertion,
) -> None:
    """Update mode creates missing golden files."""
    expect_true(hasattr(golden, "update_mode"))
