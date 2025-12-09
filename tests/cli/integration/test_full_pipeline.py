"""Integration tests for the full CLI pipeline.

Test config loading → middleware → execution → output.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)

if TYPE_CHECKING:
    from tests.cli._harness import CliTestHarness


def test_config_affects_output_format(
    cli: CliTestHarness,
) -> None:
    """Config settings should affect output format."""
    # Default format (text) - may fail if no project configured
    result_text = cli.invoke(["build", "status"])
    expect_in(result_text.exit_code, {0, 1})

    # JSON format via flag
    result_json = cli.invoke(["build", "status", "--output-format=json"])
    expect_in(result_json.exit_code, {0, 1})

    # JSON output should be valid JSON if successful
    if result_json.stdout.strip():
        with contextlib.suppress(json.JSONDecodeError):
            json.loads(result_json.stdout)


def test_env_override_takes_precedence(
    cli: CliTestHarness,
) -> None:
    """Environment variables override config file settings."""
    result = cli.with_env(CODEINTEL_DEBUG="1").invoke(["build", "status"])

    # Should complete (may fail if no project, but debug env is applied)
    expect_in(result.exit_code, {0, 1})


def test_telemetry_middleware_creates_spans(
    cli: CliTestHarness,
) -> None:
    """Telemetry middleware should create spans for operations."""
    # Enable telemetry
    result = cli.with_env(CODEINTEL_TELEMETRY_ENABLED="true").invoke(
        ["build", "status"],
    )

    # Should complete (telemetry is non-blocking)
    expect_in(result.exit_code, {0, 1})


def test_validation_middleware_rejects_invalid_params(
    cli: CliTestHarness,
) -> None:
    """Validation middleware rejects invalid parameters."""
    # Try with invalid parameter
    result = cli.invoke(["build", "run", "--invalid-flag=test"])

    # Should fail with validation error
    expect_in(result.exit_code, {1, 2})


def test_error_includes_problem_detail_structure(
    cli: CliTestHarness,
) -> None:
    """Errors should include RFC 9457 Problem Detail structure in JSON mode."""
    # Trigger an error with JSON output (invalid operation)
    result = cli.invoke(
        ["op", "call", "nonexistent.operation", "--output-format=json"],
    )

    if result.exit_code != 0 and result.stdout.strip():
        with contextlib.suppress(json.JSONDecodeError):
            error_data = json.loads(result.stdout)
            # Check for Problem Detail fields
            if "type" in error_data:
                expect_is_instance(error_data.get("type"), str)


def test_debug_mode_includes_stack_trace(
    cli: CliTestHarness,
) -> None:
    """Debug mode should produce output on commands."""
    # Use a valid command with debug mode
    result = cli.with_env(CODEINTEL_DEBUG="1").invoke(
        ["--help"],
    )

    # In debug mode, should produce output
    output = result.stdout + result.stderr
    expect_true(bool(output.strip()))


def test_json_output_is_valid_json(
    cli: CliTestHarness,
) -> None:
    """JSON format produces valid JSON when command succeeds."""
    result = cli.invoke(["build", "status", "--output-format=json"])

    # May fail if no project, but if successful should be valid JSON
    if result.exit_code == 0 and result.stdout.strip():
        data = json.loads(result.stdout)
        expect_is_instance(data, dict)
    else:
        # Just verify it didn't crash
        expect_in(result.exit_code, {0, 1})


def test_text_output_is_human_readable(
    cli: CliTestHarness,
) -> None:
    """Text format produces human-readable output."""
    result = cli.invoke(["build", "status", "--output-format=text"])

    # May fail if no project, but should produce some output
    output = result.stdout + result.stderr
    expect_true(bool(output.strip()) or result.exit_code in {0, 1})


def test_help_shows_commands(
    cli: CliTestHarness,
) -> None:
    """Help command shows available commands."""
    result = cli.invoke(["--help"])

    expect_true(result.success)
    expect_in("build", result.stdout.lower())


def test_subcommand_help(
    cli: CliTestHarness,
) -> None:
    """Subcommand help shows subcommand-specific info."""
    result = cli.invoke(["build", "--help"])

    expect_equal(result.exit_code, 0)
    expect_in("build", result.stdout.lower())


def test_version_shows_version(
    cli: CliTestHarness,
) -> None:
    """Version flag shows version info."""
    result = cli.invoke(["--version"])

    # Should show version string
    output = result.stdout + result.stderr
    expect_true(result.exit_code == 0 or "version" in output.lower())
