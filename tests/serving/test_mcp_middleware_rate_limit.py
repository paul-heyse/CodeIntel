"""Tests for FastMCP middleware behavior on the serving MCP surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp.client import Client
from mcp import McpError

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions import assert_target_ok
from tests._helpers.harnesses.serving_harness import ServingTargetHarness

if TYPE_CHECKING:
    from pathlib import Path


def _publish_serving_snapshot(serving_target_harness: ServingTargetHarness) -> Path:
    records = serving_target_harness.run_targets()
    assert_target_ok(records["serving_artifacts"])
    serving_target_harness.publish_snapshot(run_id="run-1")
    serve_dir = serving_target_harness.harness.ctx.build_paths.build_dir / "serving"
    return serve_dir / "current.json"


@pytest.mark.anyio
async def test_mcp_rate_limiting_applies_to_list_tools(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Apply rate limiting to FastMCP list_tools for a single session."""
    pointer_path = _publish_serving_snapshot(serving_target_harness)
    serve_dir = pointer_path.parent

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=serve_dir,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            mcp_rate_limit_rps=0.001,
            mcp_rate_limit_burst=1,
            mcp_cache_listings=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            await client.list_tools()
            with pytest.raises(McpError):
                await client.list_tools()
    finally:
        await manager.stop()
