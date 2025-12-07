"""Smoke tests for serving.mcp.tools exports."""

from __future__ import annotations

from codeintel.serving.mcp import tools


def test_tools_exports_serving_config_and_mode() -> None:
    """Ensure tools module re-exports serving configuration types."""
    assert hasattr(tools, "ServingConfig")
    assert hasattr(tools, "ServingMode")
