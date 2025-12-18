"""Tests for the serving DB manager hot-swap behavior."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _PointerPaths:
    semantic_registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path


def _write_registry(path: Path) -> None:
    path.write_text('{"version": "v1", "views": []}\n', encoding="utf-8")


def _write_schema_manifest(path: Path) -> None:
    path.write_text('{"version": "v1", "tables": []}\n', encoding="utf-8")


def _write_buildspec(path: Path) -> None:
    path.write_text('{"spec_version": 1, "targets": [], "datasets": []}\n', encoding="utf-8")


def _write_pointer(
    path: Path,
    *,
    db_path: Path,
    run_id: str,
    paths: _PointerPaths,
) -> None:
    payload = {
        "db_path": str(db_path),
        "semantic_registry_path": str(paths.semantic_registry_path),
        "schema_manifest_path": str(paths.schema_manifest_path),
        "buildspec_path": str(paths.buildspec_path),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": run_id,
        "published_at": datetime.now(tz=UTC).isoformat(),
        "semantic_layer_version": "v123",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_db(path: Path, *, value: int) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE kv (value INTEGER)")
    con.execute("INSERT INTO kv VALUES (?)", [value])
    con.close()


@pytest.mark.anyio
async def test_manager_initial_load_and_connect(tmp_path: Path) -> None:
    """Manager loads pointer and yields connections."""
    db1 = tmp_path / "db1.duckdb"
    _make_db(db1, value=1)

    pointer_path = tmp_path / "current.json"
    paths = _PointerPaths(
        semantic_registry_path=tmp_path / "semantic_registry.json",
        schema_manifest_path=tmp_path / "schema_manifest.json",
        buildspec_path=tmp_path / "buildspec.json",
    )
    _write_registry(paths.semantic_registry_path)
    _write_schema_manifest(paths.schema_manifest_path)
    _write_buildspec(paths.buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db1,
        run_id="run-1",
        paths=paths,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=2),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        expect_equal(manager.current_pointer().run_id, "run-1")
        with manager.connect() as (warehouse, _pointer):
            result = warehouse.gateway.con.execute("SELECT value FROM kv").fetchone()
            expect_equal(result, (1,))
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_manager_hot_swap_on_pointer_change(tmp_path: Path) -> None:
    """Pointer update swaps pools and starts reading from new snapshot DB."""
    db1 = tmp_path / "db1.duckdb"
    db2 = tmp_path / "db2.duckdb"
    _make_db(db1, value=1)
    _make_db(db2, value=2)

    pointer_path = tmp_path / "current.json"
    paths = _PointerPaths(
        semantic_registry_path=tmp_path / "semantic_registry.json",
        schema_manifest_path=tmp_path / "schema_manifest.json",
        buildspec_path=tmp_path / "buildspec.json",
    )
    _write_registry(paths.semantic_registry_path)
    _write_schema_manifest(paths.schema_manifest_path)
    _write_buildspec(paths.buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db1,
        run_id="run-1",
        paths=paths,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        with manager.connect() as (warehouse, _pointer):
            expect_equal(warehouse.gateway.con.execute("SELECT value FROM kv").fetchone(), (1,))

        _write_pointer(
            pointer_path,
            db_path=db2,
            run_id="run-2",
            paths=paths,
        )

        await asyncio.sleep(0.05)
        expect_equal(manager.current_pointer().run_id, "run-2")
        with manager.connect() as (warehouse, _pointer):
            expect_equal(warehouse.gateway.con.execute("SELECT value FROM kv").fetchone(), (2,))
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_manager_same_path_no_swap_optimization(tmp_path: Path) -> None:
    """When db_path is unchanged, pool is retained and pointer metadata updates."""
    db1 = tmp_path / "db1.duckdb"
    _make_db(db1, value=1)

    pointer_path = tmp_path / "current.json"
    paths = _PointerPaths(
        semantic_registry_path=tmp_path / "semantic_registry.json",
        schema_manifest_path=tmp_path / "schema_manifest.json",
        buildspec_path=tmp_path / "buildspec.json",
    )
    _write_registry(paths.semantic_registry_path)
    _write_schema_manifest(paths.schema_manifest_path)
    _write_buildspec(paths.buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db1,
        run_id="run-1",
        paths=paths,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        with manager.connect() as (warehouse1, _pointer):
            con1 = warehouse1.gateway.con
            expect_equal(con1.execute("SELECT value FROM kv").fetchone(), (1,))

        _write_pointer(
            pointer_path,
            db_path=db1,
            run_id="run-2",
            paths=paths,
        )

        await asyncio.sleep(0.05)
        expect_equal(manager.current_pointer().run_id, "run-2")
        with manager.connect() as (warehouse2, _pointer):
            con2 = warehouse2.gateway.con
            expect_true(con2 is con1)
            expect_equal(con2.execute("SELECT value FROM kv").fetchone(), (1,))
    finally:
        await manager.stop()


@pytest.mark.anyio
async def test_manager_export_pool_isolated_from_query_pool(tmp_path: Path) -> None:
    """Holding an export handle must not block normal query acquisition."""
    db1 = tmp_path / "db1.duckdb"
    _make_db(db1, value=1)

    pointer_path = tmp_path / "current.json"
    paths = _PointerPaths(
        semantic_registry_path=tmp_path / "semantic_registry.json",
        schema_manifest_path=tmp_path / "schema_manifest.json",
        buildspec_path=tmp_path / "buildspec.json",
    )
    _write_registry(paths.semantic_registry_path)
    _write_schema_manifest(paths.schema_manifest_path)
    _write_buildspec(paths.buildspec_path)
    _write_pointer(
        pointer_path,
        db_path=db1,
        run_id="run-1",
        paths=paths,
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        export_pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        with manager.connect_export() as (warehouse_export, _pointer):
            export_con_id = id(warehouse_export.gateway.con)

            def _query_from_pool() -> tuple[int, tuple[int] | None]:
                with manager.connect() as (warehouse, _pointer2):
                    value = warehouse.gateway.con.execute("SELECT value FROM kv").fetchone()
                    return id(warehouse.gateway.con), value

            query_con_id, value = await asyncio.wait_for(
                asyncio.to_thread(_query_from_pool), timeout=1.0
            )
            expect_true(query_con_id != export_con_id)
            expect_equal(value, (1,))
    finally:
        await manager.stop()
