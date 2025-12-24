"""Tests for FastMCP middleware behavior on the serving MCP surface."""

from __future__ import annotations

import pytest
from mcp import McpError

from tests._helpers.harnesses.serving_app import ServingAppHarness, ServingSettingsOverrides
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory


@pytest.mark.anyio
async def test_mcp_rate_limiting_applies_to_list_tools(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Apply rate limiting to FastMCP list_tools for a single session."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "mcp_rate_limit_rps": 0.001,
        "mcp_rate_limit_burst": 1,
        "mcp_cache_listings": False,
    }
    async with harness.mcp_client(settings_overrides=settings_overrides) as client:
        await client.list_tools()
        with pytest.raises(McpError):
            await client.list_tools()
