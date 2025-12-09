"""Integration tests for the resolution layer.

Tests verify that the resolution/ package and ExecutionContext enhancements
work correctly together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.execution.context import ExecutionContext
from codeintel.cli.options import CommonOptions
from codeintel.cli.resolution import ResolutionError
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_none,
)

# Test constants for magic value checks
EXPECTED_VERBOSITY = 2


def test_resolution_from_explicit_params() -> None:
    """Test runtime resolution from explicit params."""
    ctx = ExecutionContext.for_sync(
        "test.op",
        {
            "repo": "test/repo",
            "commit": "abc123def456789",
            "db_path": str(Path.cwd() / "build" / "test.duckdb"),
            "repo_root": str(Path.cwd()),
        },
    )

    runtime = ctx.require_runtime()
    expect_equal(runtime.repo, "test/repo")
    expect_equal(runtime.commit, "abc123def456789")


def test_resolution_missing_params_raises_error() -> None:
    """Test resolution fails with missing params."""
    ctx = ExecutionContext.for_sync("test.op", {})

    with pytest.raises(ResolutionError):
        ctx.require_runtime()


def test_context_creation() -> None:
    """Test basic context creation."""
    ctx = ExecutionContext.for_sync(
        "test.operation",
        {"verbose": 1, "target": "all"},
    )

    expect_equal(ctx.operation_id, "test.operation")
    expect_equal(ctx.verbosity, 1)
    expect_equal(ctx.get_str_param("target", "default"), "all")


def test_str_param_with_default() -> None:
    """Test get_str_param with default."""
    ctx = ExecutionContext.for_sync("test.op", {})

    # Missing param returns default
    result = ctx.get_str_param("missing", "fallback")
    expect_equal(result, "fallback")

    # Missing param with no default returns None
    result_none = ctx.get_str_param("missing")
    expect_is_none(result_none)


def test_require_str_param_raises() -> None:
    """Test require_str_param raises for missing."""
    ctx = ExecutionContext.for_sync("test.op", {})

    with pytest.raises(ValueError, match="Required parameter"):
        ctx.require_str_param("missing")


def test_context_close_is_idempotent() -> None:
    """Test close() is safe to call multiple times."""
    ctx = ExecutionContext.for_sync("test.op", {})

    # Close multiple times should be safe
    ctx.close()
    ctx.close()
    ctx.close()


def test_options_to_params() -> None:
    """Test converting CommonOptions to params dict."""
    options = CommonOptions(
        repo="test/repo",
        commit="abc123",
        verbose=EXPECTED_VERBOSITY,
    )

    params = options.to_params()

    expect_equal(params["repo"], "test/repo")
    expect_equal(params["commit"], "abc123")
    expect_equal(params["verbose"], EXPECTED_VERBOSITY)


def test_output_format_resolution() -> None:
    """Test output format resolution with json flag."""
    # JSON flag takes precedence
    options = CommonOptions(
        output_format=OutputFormat.TEXT,
        json=True,
    )
    expect_equal(options.resolve_output_format(), OutputFormat.JSON)

    # Without json flag, use output_format
    options2 = CommonOptions(
        output_format=OutputFormat.TEXT,
        json=False,
    )
    expect_equal(options2.resolve_output_format(), OutputFormat.TEXT)
