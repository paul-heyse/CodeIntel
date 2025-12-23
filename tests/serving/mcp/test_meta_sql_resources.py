"""Tests for snapshot-scoped meta SQL resources (views_sql + views_sql_diff)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import duckdb
import pytest
from fastmcp.client import Client
from mcp import McpError

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.mcp.app import build_mcp_app
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.types import TextResourceContents


def _make_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.v_demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.v_demo VALUES (1, 'one')")
    con.close()


def _write_registry(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_semantic_registry(
        path=path,
        views=[
            {
                "id": "demo.view",
                "kind": "view",
                "table_key": "docs.v_demo",
                "entity": "demo",
                "grain": "per_row",
                "description": "Demo view",
                "primary_key": ["id"],
                "columns": ["id", "label"],
                "joins": [],
                "defaults": {"limit": 200, "order_by": ["id"]},
                "sensitivity": "internal",
                "deprecated": False,
                "replaced_by": None,
            }
        ],
    )


def _write_schema_manifest(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_schema_manifest(
        path=path,
        tables=[
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            }
        ],
    )


def _write_buildspec(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_buildspec(
        path=path,
        datasets=[{"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"}],
    )


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    pointer = {
        "db_path": str(db_path),
        "semantic_registry_path": str(registry_path),
        "schema_manifest_path": str(manifest_path),
        "buildspec_path": str(buildspec_path),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": "run-1",
        "published_at": datetime.now(tz=UTC).isoformat(),
        "semantic_layer_version": "v123",
    }
    path.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")


def _setup_test_snapshot(tmp_path: Path) -> Path:
    db_path = tmp_path / "codeintel.duckdb"
    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    pointer_path = tmp_path / "current.json"

    _make_db(db_path)
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )
    return pointer_path


@pytest.mark.anyio
async def test_mcp_meta_views_sql_resources_round_trip(tmp_path: Path) -> None:
    """Expose views_sql and views_sql_diff when artifacts exist."""
    pointer_path = _setup_test_snapshot(tmp_path)

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
    pointer_path = _setup_test_snapshot(tmp_path)

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
