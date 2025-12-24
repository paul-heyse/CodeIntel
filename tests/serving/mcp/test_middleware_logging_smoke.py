"""Smoke tests for the MCP middleware stack (logging/timing/error mapping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp.client import Client

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_true
from tests._helpers.serving_snapshots import setup_demo_snapshot

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.anyio
async def test_mcp_middleware_logging_smoke(tmp_path: Path) -> None:
    """Ensure middleware stack supports list_tools + call_tool without crashing."""
    pointer_path = setup_demo_snapshot(tmp_path).pointer_path

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=tmp_path,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            mcp_enable_structured_logging=True,
            mcp_cache_listings=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            tools = await client.list_tools()
            expect_true(any(t.name == "semantic_catalog" for t in tools))
            _ = await client.call_tool("semantic_catalog", {})
    finally:
        await manager.stop()
