"""Tests for tools module exports."""

from __future__ import annotations

from codeintel.serving.mcp import tools


def test_tools_exports_serving_config_and_mode() -> None:
    """Module should expose ServingConfig and ServingMode."""
    assert hasattr(tools, "ServingConfig")
    assert hasattr(tools, "ServingMode")
