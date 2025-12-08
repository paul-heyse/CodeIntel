"""Tests for tools module exports."""

from __future__ import annotations

from codeintel.serving.mcp import tools
from tests._helpers.assertions import expect_true


def test_tools_exports_serving_config_and_mode() -> None:
    """Module should expose ServingConfig and ServingMode."""
    expect_true(hasattr(tools, "ServingConfig"))
    expect_true(hasattr(tools, "ServingMode"))
