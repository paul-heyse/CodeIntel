"""Tests for MCP server creation and lifespan management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.server import create_mcp_server
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
)

if TYPE_CHECKING:
    from pathlib import Path


class _CountingDBManager(ServingDBManager):
    """ServingDBManager that records start calls."""

    start_calls: int = 0

    async def start(self) -> None:
        self.start_calls += 1


def test_create_mcp_server_creates_own_db_manager(tmp_path: Path) -> None:
    """Verify standalone server creates and manages its own db_manager."""
    settings = ServingSettings(serve_dir=tmp_path)
    mcp = create_mcp_server(settings)
    # Server should be created successfully
    expect_is_not_none(mcp)
    expect_equal(mcp.name, "CodeIntel")


def test_create_mcp_server_accepts_injected_db_manager(tmp_path: Path) -> None:
    """Verify server accepts pre-configured db_manager."""
    settings = ServingSettings(serve_dir=tmp_path)
    manager = _CountingDBManager(pointer_path=tmp_path / "current.json")

    mcp = create_mcp_server(settings, db_manager=manager)
    expect_is_not_none(mcp)
    expect_equal(mcp.name, "CodeIntel")


def test_create_mcp_server_does_not_start_injected_db_manager(tmp_path: Path) -> None:
    """Verify injected db_manager lifecycle is not managed by MCP server."""
    settings = ServingSettings(serve_dir=tmp_path)
    manager = _CountingDBManager(pointer_path=tmp_path / "current.json")

    # Creating the server should not call start() on injected db_manager
    mcp = create_mcp_server(settings, db_manager=manager)
    expect_is_not_none(mcp)
    # The db_manager.start() should not be called during server creation
    # (it would only be called during lifespan if we owned it)
    expect_equal(manager.start_calls, 0)


def test_create_mcp_server_with_default_settings(tmp_path: Path) -> None:
    """Verify server can be created with only serve_dir specified."""
    settings = ServingSettings(serve_dir=tmp_path)
    mcp = create_mcp_server(settings)
    expect_is_not_none(mcp)
    expect_equal(mcp.name, "CodeIntel")
