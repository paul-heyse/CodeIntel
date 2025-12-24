"""PR-87: runtime semantic queries enforce allowed columns against schema manifest."""

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
        gateway.con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two')")
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
            "columns": ["id", "label", "bogus"],
            "defaults": {"limit": 10, "order_by": []},
            "joins": [],
            "primary_key": ["id"],
            "description": None,
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
async def test_strict_mode_rejects_unknown_columns(tmp_path: Path) -> None:
    """Strict schema enforcement rejects unknown semantic view columns."""
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
                result_engine="pandas",
            ),
        )
        with pytest.raises(ValueError, match=r"exposes unknown columns"):
            kernel.query(SemanticQueryRequest(view_id="demo.view"))
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_warn_mode_intersects_unknown_columns(tmp_path: Path) -> None:
    """Warn schema enforcement intersects unknown columns with the manifest."""
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
                schema_enforcement="warn",
                result_engine="pandas",
            ),
        )
        res = kernel.query(SemanticQueryRequest(view_id="demo.view"))
        expect_equal(res.columns, ["id", "label"])
    finally:
        await manager.stop()
