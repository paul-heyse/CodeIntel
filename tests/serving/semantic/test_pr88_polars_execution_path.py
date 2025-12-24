"""PR-88: serving query execution can use DuckDB -> Polars extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.gateway import GatewayFactory
from tests._helpers.schemas import ensure_production_schemas
from tests._helpers.serving_snapshot_factory import (
    ServingSnapshotFactory,
    SnapshotArtifacts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.serving_snapshot_factory import ServingSnapshot


def _make_snapshot_db(db_path: Path) -> None:
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        ensure_production_schemas(gateway.con)
        gateway.con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
        gateway.con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
        gateway.con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    finally:
        gateway.close()


def _registry_views() -> list[dict[str, object]]:
    return [
        {
            "id": "demo.view",
            "kind": "view",
            "table_key": "docs.v_demo",
            "entity": "demo",
            "grain": "per_row",
            "description": None,
            "primary_key": ["id"],
            "columns": ["id", "label"],
            "joins": [],
            "defaults": {"limit": 200, "order_by": ["id"]},
            "sensitivity": "internal",
        }
    ]


def _manifest_tables() -> list[dict[str, object]]:
    return [
        {
            "schema": "docs",
            "name": "v_demo",
            "table_key": "docs.v_demo",
            "description": None,
            "primary_key": ["id"],
            "indexes": [],
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "label", "type": "VARCHAR", "nullable": True},
            ],
        }
    ]


def _make_snapshot(factory: ServingSnapshotFactory) -> ServingSnapshot:
    artifacts = SnapshotArtifacts(
        views=_registry_views(),
        tables=_manifest_tables(),
        db_setup=_make_snapshot_db,
    )
    return factory.make_snapshot(artifacts=artifacts)


@pytest.mark.anyio
async def test_polars_execution_path_matches_expected_rows(tmp_path: Path) -> None:
    """Polars result extraction returns expected rows."""
    pytest.importorskip("polars")
    snapshot = _make_snapshot(ServingSnapshotFactory(tmp_path))
    manager = ServingDBManager(pointer_path=snapshot.pointer_path, pool_cfg=PoolConfig(size=1))
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                schema_enforcement="strict",
                result_engine="polars",
            ),
        )
        res = kernel.query(SemanticQueryRequest(view_id="demo.view", limit=10))
        expect_equal([row["id"] for row in res.rows], [1, 2, 3])
    finally:
        await manager.stop()
