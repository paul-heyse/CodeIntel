"""Tests for MCP tool feature flags."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.settings import ServingSettings, get_serving_settings
from tests._helpers.assertions.expectation_assertions import (
    expect_false,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures("codeintel_env")


def test_default_all_tools_enabled(tmp_path: Path) -> None:
    """Verify all MCP tools are enabled by default."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_search_via_env(tmp_path: Path) -> None:
    """Verify code_search can be disabled via environment variable."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "0",
        }
    )
    settings = get_serving_settings()
    expect_false(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_explain_via_env(tmp_path: Path) -> None:
    """Verify semantic_explain can be disabled via environment variable."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_EXPLAIN": "0",
        }
    )
    settings = get_serving_settings()
    expect_true(settings.mcp_enable_search)
    expect_false(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_meta_via_env(tmp_path: Path) -> None:
    """Verify serving_meta can be disabled via environment variable."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_META": "0",
        }
    )
    settings = get_serving_settings()
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_false(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_export_via_env(tmp_path: Path) -> None:
    """Verify semantic_export can be disabled via environment variable."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_EXPORT": "0",
        }
    )
    settings = get_serving_settings()
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_false(settings.mcp_enable_export)


def test_disable_multiple_tools_via_env(tmp_path: Path) -> None:
    """Verify multiple tools can be disabled simultaneously."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "0",
            "CODEINTEL_MCP_ENABLE_META": "0",
        }
    )
    settings = get_serving_settings()
    expect_false(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_false(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_enable_search_explicitly_via_env(tmp_path: Path) -> None:
    """Verify code_search can be explicitly enabled via env."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "1",
        }
    )
    settings = get_serving_settings()
    expect_true(settings.mcp_enable_search)


def test_enable_all_explicitly_via_env(tmp_path: Path) -> None:
    """Verify all tools can be explicitly enabled via env."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "1",
            "CODEINTEL_MCP_ENABLE_EXPLAIN": "1",
            "CODEINTEL_MCP_ENABLE_META": "1",
            "CODEINTEL_MCP_ENABLE_EXPORT": "1",
        }
    )
    settings = get_serving_settings()
    expect_true(settings.mcp_enable_search)
    expect_true(settings.mcp_enable_explain)
    expect_true(settings.mcp_enable_meta)
    expect_true(settings.mcp_enable_export)


def test_disable_all_optional_tools(tmp_path: Path) -> None:
    """Verify all optional tools can be disabled at once."""
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path),
            "CODEINTEL_MCP_ENABLE_SEARCH": "0",
            "CODEINTEL_MCP_ENABLE_EXPLAIN": "0",
            "CODEINTEL_MCP_ENABLE_META": "0",
            "CODEINTEL_MCP_ENABLE_EXPORT": "0",
        }
    )
    settings = get_serving_settings()
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
