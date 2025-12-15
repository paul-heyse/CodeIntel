"""Integration tests for the semantic query kernel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pool import DuckDBPoolConfig
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import FilterSpec, SemanticQueryRequest
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from pathlib import Path


def _make_snapshot_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    con.close()


def _write_registry(path: Path) -> None:
    registry = {
        "version": "v1",
        "views": [
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
                "defaults": {"limit": 2, "order_by": ["id"]},
                "sensitivity": "internal",
                "deprecated": False,
                "replaced_by": None,
            }
        ],
    }
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def _write_schema_manifest(path: Path) -> None:
    manifest = {
        "version": "v1",
        "tables": [
            {
                "schema": "docs",
                "name": "demo",
                "table_key": "docs.demo",
                "description": "Demo table schema",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            },
            {
                "schema": "docs",
                "name": "v_demo",
                "table_key": "docs.v_demo",
                "description": "Demo view schema",
                "primary_key": ["id"],
                "indexes": [],
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "label", "type": "VARCHAR", "nullable": True},
                ],
            },
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
) -> None:
    payload = {
        "db_path": str(db_path),
        "semantic_registry_path": str(registry_path),
        "schema_manifest_path": str(manifest_path),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": "run-1",
        "published_at": datetime.now(tz=UTC).isoformat(),
        "semantic_layer_version": "v123",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.anyio
async def test_kernel_catalog_describe_query_meta(tmp_path: Path) -> None:
    """Kernel exposes catalog/describe/query/meta over the current snapshot."""
    db_path = tmp_path / "codeintel.duckdb"
    _make_snapshot_db(db_path)

    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)

    pointer_path = tmp_path / "current.json"
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=DuckDBPoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(db=manager)

        catalog = kernel.catalog()
        expect_equal(catalog.get("version"), "v1")
        views_raw = catalog.get("views")
        if not isinstance(views_raw, list):
            pytest.fail("Expected catalog.views to be a list")
        expect_true(any(isinstance(v, dict) and v.get("id") == "demo.view" for v in views_raw))

        desc = kernel.describe("demo.view")
        expect_equal(desc.get("table_key"), "docs.v_demo")
        column_types_raw = desc.get("column_types")
        if not isinstance(column_types_raw, dict):
            pytest.fail("Expected desc.column_types to be a dict")
        expect_equal(column_types_raw.get("id"), "INTEGER")

        req = SemanticQueryRequest(
            view_id="demo.view",
            filters=[FilterSpec(column="id", op="gte", value=2)],
            order_by=["-id"],
            limit=10,
            offset=0,
        )
        res = kernel.query(req)
        expect_equal(res.columns, ["id", "label"])
        expect_equal([row["id"] for row in res.rows], [3, 2])

        meta = kernel.meta()
        expect_equal(meta["schema_inventory"], {"tables": 1, "views": 1})
    finally:
        await manager.stop()
