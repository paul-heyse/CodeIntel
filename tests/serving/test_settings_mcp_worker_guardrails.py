"""Tests for enforcing the sessionful MCP single-worker contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from pathlib import Path


def test_create_serving_app_rejects_multi_worker_when_mcp_mounted(tmp_path: Path) -> None:
    """Reject multi-worker uvicorn when sessionful MCP is mounted."""
    settings = ServingSettings(serve_dir=tmp_path, uvicorn_workers=2)
    with pytest.raises(ValueError, match="uvicorn_workers=1"):
        create_serving_app(settings=settings, mount_mcp=True)


def test_create_serving_app_allows_multi_worker_when_mcp_not_mounted(tmp_path: Path) -> None:
    """Allow multi-worker uvicorn when MCP is not mounted."""
    settings = ServingSettings(serve_dir=tmp_path, uvicorn_workers=2)
    create_serving_app(settings=settings, mount_mcp=False)
