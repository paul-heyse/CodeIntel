"""Tests for MCP tool feature flags."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import (
    expect_false,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextlib.contextmanager
def _set_env(env: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables.

    Parameters
    ----------
    env
        Environment variables to set.

    Yields
    ------
    None
        Context manager scope.
    """
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_default_all_tools_enabled(tmp_path: Path) -> None:
    """Verify all MCP tools are enabled by default."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_search_via_env(tmp_path: Path) -> None:
    """Verify code_search can be disabled via environment variable."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "0",
        }
    ):
        settings = ServingSettings.from_env()
    expect_false(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_explain_via_env(tmp_path: Path) -> None:
    """Verify semantic_explain can be disabled via environment variable."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_EXPLAIN": "0",
        }
    ):
        settings = ServingSettings.from_env()
    expect_true(settings.mcp_enable_search)
    expect_false(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_meta_via_env(tmp_path: Path) -> None:
    """Verify serving_meta can be disabled via environment variable."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_META": "0",
        }
    ):
        settings = ServingSettings.from_env()
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_false(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_export_via_env(tmp_path: Path) -> None:
    """Verify semantic_export can be disabled via environment variable."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_EXPORT": "0",
        }
    ):
        settings = ServingSettings.from_env()
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_false(settings.mcp_enable_export)


def test_disable_multiple_tools_via_env(tmp_path: Path) -> None:
    """Verify multiple tools can be disabled simultaneously."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "0",
            "CODEINTEL_MCP_ENABLE_META": "0",
        }
    ):
        settings = ServingSettings.from_env()
    expect_false(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_false(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_enable_search_explicitly_via_env(tmp_path: Path) -> None:
    """Verify code_search can be explicitly enabled via env."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "1",
        }
    ):
        settings = ServingSettings.from_env()
    expect_true(settings.mcp_enable_search)


def test_enable_all_explicitly_via_env(tmp_path: Path) -> None:
    """Verify all tools can be explicitly enabled via env."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "1",
            "CODEINTEL_MCP_ENABLE_EXPLAIN": "1",
            "CODEINTEL_MCP_ENABLE_META": "1",
            "CODEINTEL_MCP_ENABLE_EXPORT": "1",
        }
    ):
        settings = ServingSettings.from_env()
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_all_optional_tools(tmp_path: Path) -> None:
    """Verify all optional tools can be disabled at once."""
    with _set_env(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "0",
            "CODEINTEL_MCP_ENABLE_EXPLAIN": "0",
            "CODEINTEL_MCP_ENABLE_META": "0",
            "CODEINTEL_MCP_ENABLE_EXPORT": "0",
        }
    ):
        settings = ServingSettings.from_env()
    expect_false(settings.mcp_enable_search)
    expect_false(settings.mcp_enable_explain)
    expect_false(settings.mcp_enable_meta)
    expect_false(settings.mcp_enable_export)


def test_constructor_override(tmp_path: Path) -> None:
    """Verify feature flags can be set via constructor."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        mcp_enable_search=False,
        mcp_enable_explain=False,
        mcp_enable_meta=True,
        mcp_enable_export=True,
    )
    expect_false(settings.mcp_enable_search)
    expect_false(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)
