"""Integration tests for the semantic query kernel."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import (
    FilterSpec,
    SemanticExportRequest,
    SemanticQueryRequest,
)
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.gateway import GatewayFactory
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from pathlib import Path


def _make_snapshot_db(db_path: Path) -> None:
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        ensure_production_schemas(gateway.con)
        gateway.con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
        gateway.con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
        gateway.con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    finally:
        gateway.close()


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
                "defaults": {"limit": 2, "order_by": ["id"]},
                "sensitivity": "internal",
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
    )


def _write_buildspec(path: Path) -> None:
    artifacts = HarnessArtifacts(
        repo_root=path.parent,
        paths=BuildPaths.from_explicit(build_dir=path.parent),
    )
    artifacts.write_buildspec(
        path=path,
        datasets=[
            {"table_key": "docs.demo", "schema_hash": "schema_demo"},
            {"table_key": "docs.v_demo", "schema_hash": "schema_v_demo"},
        ],
    )


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    registry_path: Path,
    manifest_path: Path,
    buildspec_path: Path,
) -> None:
    pointer = ServingSnapshotPointer(
        db_path=db_path,
        semantic_registry_path=registry_path,
        schema_manifest_path=manifest_path,
        buildspec_path=buildspec_path,
        repo="demo/repo",
        commit="deadbeef",
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v123",
    )
    path.write_text(pointer.to_json(), encoding="utf-8")


@pytest.mark.anyio
async def test_kernel_catalog_describe_query_meta(tmp_path: Path) -> None:
    """Kernel exposes catalog/describe/query/meta over the current snapshot."""
    db_path = tmp_path / "codeintel.duckdb"
    _make_snapshot_db(db_path)

    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)

    pointer_path = tmp_path / "current.json"
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=tmp_path,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="pandas",
                schema_enforcement="strict",
            ),
        )

        catalog = kernel.catalog()
        expect_equal(catalog.version, "v1")
        expect_true(any(view.id == "demo.view" for view in catalog.views))

        desc = kernel.describe("demo.view")
        expect_equal(desc.table_key, "docs.v_demo")
        expect_equal(desc.column_types.get("id"), "INTEGER")

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
        expect_equal(meta.schema_inventory, {"tables": 1, "views": 1})
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_kernel_describe_includes_lineage(tmp_path: Path) -> None:
    """Describe includes column lineage when metadata is present."""
    db_path = tmp_path / "codeintel.duckdb"
    _make_snapshot_db(db_path)

    con = duckdb.connect(str(db_path))
    try:
        ensure_production_schemas(con)
        con.execute(
            """
            INSERT INTO metadata.derived_lineage_columns (
                repo,
                commit,
                downstream_table,
                downstream_column,
                upstream_table,
                upstream_column,
                edge_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "demo/repo",
                "deadbeef",
                "docs.v_demo",
                "label",
                "docs.demo",
                "label",
                "derived_column_depends_on",
            ],
        )
    finally:
        con.close()

    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)

    pointer_path = tmp_path / "current.json"
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=tmp_path,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="pandas",
                schema_enforcement="strict",
            ),
        )

        desc = kernel.describe("demo.view")
        expect_true("label" in desc.lineage, message="lineage entry present")
        expect_equal(desc.lineage["label"][0].table_key, "docs.demo")
        expect_equal(desc.lineage["label"][0].column, "label")
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_kernel_export_rows_close_releases_export_pool(tmp_path: Path) -> None:
    """Closing an export generator must release the export connection."""
    db_path = tmp_path / "codeintel.duckdb"
    _make_snapshot_db(db_path)

    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    _write_registry(registry_path)
    _write_schema_manifest(manifest_path)
    _write_buildspec(buildspec_path)

    pointer_path = tmp_path / "current.json"
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        export_pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
        hot_swap=False,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=tmp_path,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="pandas",
                schema_enforcement="strict",
            ),
        )

        request = SemanticExportRequest(
            view_id="demo.view",
            order_by=["id"],
            limit=100,
            offset=0,
        )
        gen = kernel.export_rows(request)
        first = next(gen)
        expect_equal(first["id"], 1)
        gen.close()

        def _acquire_export_and_fetch() -> tuple[int, tuple[int] | None]:
            with manager.connect_export() as (warehouse, _pointer2):
                return id(warehouse.gateway.con), warehouse.gateway.con.execute(
                    "SELECT COUNT(*) FROM docs.demo"
                ).fetchone()

        _con_id, count_row = await asyncio.wait_for(
            asyncio.to_thread(_acquire_export_and_fetch),
            timeout=1.0,
        )
        expect_equal(count_row, (3,))
    finally:
        await manager.stop()
