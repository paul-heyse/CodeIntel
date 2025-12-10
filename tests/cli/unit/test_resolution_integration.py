"""Integration tests for the resolution layer.

Tests verify that the resolution/ package and HandlerContext work correctly together.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeintel.cli.commands._common import CommonOptions
from codeintel.cli.handlers.context import HandlerContext
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.resolution import ResolutionError, resolve_from_params
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_none,
)

# Test constants for magic value checks
EXPECTED_VERBOSITY = 2


def test_resolution_from_explicit_params() -> None:
    """Test runtime resolution from explicit params."""
    params = {
        "repo": "test/repo",
        "commit": "abc123def456789",
        "db_path": str(Path.cwd() / "build" / "test.duckdb"),
        "repo_root": str(Path.cwd()),
    }

    runtime = resolve_from_params(params)
    expect_equal(runtime.repo, "test/repo")
    expect_equal(runtime.commit, "abc123def456789")


def test_resolution_missing_params_raises_error() -> None:
    """Test resolution fails with missing params."""
    with pytest.raises(ResolutionError):
        resolve_from_params({})


def test_handler_context_creation() -> None:
    """Test basic HandlerContext creation."""
    config = MagicMock()
    config.log_level = "WARNING"
    ctx = HandlerContext(
        config=config,
        operation_id="test.operation",
        _params={"verbose": 1, "target": "all"},
    )

    expect_equal(ctx.operation_id, "test.operation")
    expect_equal(ctx.param_str("target", "default"), "all")


def test_handler_context_str_param_with_default() -> None:
    """Test param_str with default."""
    config = MagicMock()
    ctx = HandlerContext(
        config=config,
        operation_id="test.op",
        _params={},
    )

    # Missing param returns default
    result = ctx.param_str("missing", "fallback")
    expect_equal(result, "fallback")

    # Missing param with no default returns None
    result_none = ctx.param_str("missing")
    expect_is_none(result_none)


def test_handler_context_close_is_idempotent() -> None:
    """Test close() is safe to call multiple times."""
    config = MagicMock()
    ctx = HandlerContext(
        config=config,
        operation_id="test.op",
        _params={},
    )

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
