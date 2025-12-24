"""PR-87: runtime semantic queries enforce allowed columns against schema manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.gateway import GatewayFactory
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _make_snapshot_db(db_path: Path) -> None:
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        ensure_production_schemas(gateway.con)
        gateway.con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
        gateway.con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two')")
        gateway.con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    finally:
        gateway.close()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
async def test_strict_mode_rejects_unknown_columns(tmp_path: Path) -> None:
    """Strict schema enforcement rejects unknown semantic view columns."""
    db_path = tmp_path / "codeintel.duckdb"
    _make_snapshot_db(db_path)

    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    pointer_path = tmp_path / "current.json"

    _write_json(
        registry_path,
        {
            "version": "v1",
            "views": [
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
            ],
        },
    )
    _write_json(
        manifest_path,
        {
            "version": "v2",
            "tables": [],
            "views": [
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
            ],
        },
    )
    _write_json(
        buildspec_path,
        {
            "spec_version": 1,
            "targets": [],
            "datasets": [{"table_key": "docs.v_demo", "schema_hash": "h"}],
        },
    )
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    manager = ServingDBManager(pointer_path=pointer_path, pool_cfg=PoolConfig(size=1))
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=tmp_path,
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
    db_path = tmp_path / "codeintel.duckdb"
    _make_snapshot_db(db_path)

    registry_path = tmp_path / "semantic_registry.json"
    manifest_path = tmp_path / "schema_manifest.json"
    buildspec_path = tmp_path / "buildspec.json"
    pointer_path = tmp_path / "current.json"

    _write_json(
        registry_path,
        {
            "version": "v1",
            "views": [
                {
                    "id": "demo.view",
                    "kind": "view",
                    "table_key": "docs.v_demo",
                    "entity": "demo",
                    "grain": "per_row",
                    "columns": ["id", "label", "bogus"],
                    "defaults": {"limit": 10, "order_by": ["id"]},
                    "joins": [],
                    "primary_key": ["id"],
                    "description": None,
                    "sensitivity": "internal",
                }
            ],
        },
    )
    _write_json(
        manifest_path,
        {
            "version": "v2",
            "tables": [],
            "views": [
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
            ],
        },
    )
    _write_json(
        buildspec_path,
        {
            "spec_version": 1,
            "targets": [],
            "datasets": [{"table_key": "docs.v_demo", "schema_hash": "h"}],
        },
    )
    _write_pointer(
        pointer_path,
        db_path=db_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        buildspec_path=buildspec_path,
    )

    manager = ServingDBManager(pointer_path=pointer_path, pool_cfg=PoolConfig(size=1))
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=tmp_path,
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
