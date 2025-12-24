"""Tests for snapshot-scoped meta SQL resources (views_sql + views_sql_diff)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
from mcp import McpError

from tests._helpers.harnesses.serving_app import ServingAppHarness
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from mcp.types import TextResourceContents


@pytest.mark.anyio
async def test_mcp_meta_views_sql_resources_round_trip(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Expose views_sql and views_sql_diff when artifacts exist."""
    snapshot = serving_snapshot_factory.demo_snapshot(row_count=1)
    views_sql_path = snapshot.serve_dir / "views_sql.json"
    views_sql_diff_path = snapshot.serve_dir / "views_sql_diff.json"
    views_sql_path.write_text(
        json.dumps({"demo.view": "SELECT 1 AS one"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    views_sql_diff_path.write_text(
        json.dumps({"demo.view": {"changed": True}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    harness = ServingAppHarness.from_snapshot(snapshot)
    async with harness.mcp_client(settings_overrides={"hot_swap": False}) as client:
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


@pytest.mark.anyio
async def test_mcp_meta_views_sql_rejects_unsafe_sql(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Reject non-select SQL payloads in views_sql.json."""
    snapshot = serving_snapshot_factory.demo_snapshot(row_count=1)
    (snapshot.serve_dir / "views_sql.json").write_text(
        json.dumps({"demo.view": "DROP TABLE docs.v_demo"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    harness = ServingAppHarness.from_snapshot(snapshot)
    async with harness.mcp_client(settings_overrides={"hot_swap": False}) as client:
        with pytest.raises(McpError):
            await client.read_resource("codeintel://meta/views_sql")
