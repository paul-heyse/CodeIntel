"""Integration tests for the resolution layer.

Tests verify that the resolution/ package works correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli.resolution import ResolutionError, resolve_from_params
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_resolution_from_explicit_params() -> None:
    """Test runtime resolution from explicit params."""
    params: dict[str, object | str] = {
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
