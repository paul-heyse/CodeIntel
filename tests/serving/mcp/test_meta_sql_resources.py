"""Tests for snapshot-scoped meta SQL resources (views_sql + views_sql_diff)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
from fastmcp.client import Client
from mcp import McpError

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import TextResourceContents


def _setup_demo_snapshot(tmp_path: Path) -> Path:
    snapshot = ServingSnapshotFactory(tmp_path, serve_dir=tmp_path).demo_snapshot(row_count=1)
    return snapshot.pointer_path


@pytest.mark.anyio
async def test_mcp_meta_views_sql_resources_round_trip(tmp_path: Path) -> None:
    """Expose views_sql and views_sql_diff when artifacts exist."""
    pointer_path = _setup_demo_snapshot(tmp_path)

    views_sql_path = tmp_path / "views_sql.json"
    views_sql_diff_path = tmp_path / "views_sql_diff.json"
    views_sql_path.write_text(
        json.dumps({"demo.view": "SELECT 1 AS one"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    views_sql_diff_path.write_text(
        json.dumps({"demo.view": {"changed": True}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            views_sql = await client.read_resource("codeintel://meta/views_sql")
            content_item = cast("TextResourceContents", views_sql[0])
            payload = json.loads(content_item.text)
            if payload.get("demo.view") != "SELECT 1 AS one":
                pytest.fail("Expected views_sql payload to include demo.view compiled SQL")

            views_sql_diff = await client.read_resource("codeintel://meta/views_sql_diff")
            diff_item = cast("TextResourceContents", views_sql_diff[0])
            diff_payload = json.loads(diff_item.text)
            if diff_payload.get("demo.view") is None:
                pytest.fail("Expected views_sql_diff payload to include demo.view diff entry")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_mcp_meta_views_sql_rejects_unsafe_sql(tmp_path: Path) -> None:
    """Reject non-select SQL payloads in views_sql.json."""
    pointer_path = _setup_demo_snapshot(tmp_path)

    (tmp_path / "views_sql.json").write_text(
        json.dumps({"demo.view": "DROP TABLE docs.v_demo"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            with pytest.raises(McpError):
                await client.read_resource("codeintel://meta/views_sql")
    finally:
        await manager.stop()
