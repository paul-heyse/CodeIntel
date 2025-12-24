"""Smoke tests for the MCP middleware stack (logging/timing/error mapping)."""

from __future__ import annotations

import pytest

from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.harnesses.serving_app import ServingAppHarness, ServingSettingsOverrides
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory


@pytest.mark.anyio
async def test_mcp_middleware_logging_smoke(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Ensure middleware stack supports list_tools + call_tool without crashing."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "mcp_enable_structured_logging": True,
        "mcp_cache_listings": False,
    }
    async with harness.mcp_client(settings_overrides=settings_overrides) as client:
        tools = await client.list_tools()
        expect_true(any(t.name == "semantic_catalog" for t in tools))
        _ = await client.call_tool("semantic_catalog", {})
