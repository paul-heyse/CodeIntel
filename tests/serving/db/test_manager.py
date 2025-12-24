"""Tests for the serving DB manager hot-swap behavior."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.serving_snapshot_factory import (
    ServingSnapshotFactory,
    SnapshotArtifacts,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.serving_snapshot_factory import ServingSnapshot


def _make_db(path: Path, *, value: int) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE kv (value INTEGER)")
    con.execute("INSERT INTO kv VALUES (?)", [value])
    con.close()


def _make_snapshot(
    factory: ServingSnapshotFactory,
    *,
    run_id: str,
    value: int,
) -> ServingSnapshot:
    artifacts = SnapshotArtifacts(
        views=[],
        tables=[],
        db_setup=lambda db_path: _make_db(db_path, value=value),
    )
    return factory.make_snapshot(run_id=run_id, artifacts=artifacts)


@pytest.mark.anyio
async def test_manager_initial_load_and_connect(tmp_path: Path) -> None:
    """Manager loads pointer and yields connections."""
    snapshot = _make_snapshot(
        ServingSnapshotFactory(tmp_path, serve_dir=tmp_path / "snapshot-1"),
        run_id="run-1",
        value=1,
    )

    manager = ServingDBManager(
        pointer_path=snapshot.pointer_path,
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
    snapshot1 = _make_snapshot(
        ServingSnapshotFactory(tmp_path, serve_dir=tmp_path / "snapshot-1"),
        run_id="run-1",
        value=1,
    )
    snapshot2 = _make_snapshot(
        ServingSnapshotFactory(tmp_path, serve_dir=tmp_path / "snapshot-2"),
        run_id="run-2",
        value=2,
    )
    pointer_path = snapshot1.pointer_path

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        with manager.connect() as (warehouse, _pointer):
            expect_equal(warehouse.gateway.con.execute("SELECT value FROM kv").fetchone(), (1,))

        pointer_path.write_text(
            snapshot2.pointer_path.read_text(encoding="utf-8"),
            encoding="utf-8",
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
    snapshot = _make_snapshot(
        ServingSnapshotFactory(tmp_path, serve_dir=tmp_path / "snapshot-1"),
        run_id="run-1",
        value=1,
    )
    pointer_path = snapshot.pointer_path

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

        updated = replace(
            ServingSnapshotPointer.load(pointer_path),
            run_id="run-2",
            published_at=datetime.now(tz=UTC),
        )
        pointer_path.write_text(updated.to_json(), encoding="utf-8")

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
    snapshot = _make_snapshot(
        ServingSnapshotFactory(tmp_path, serve_dir=tmp_path / "snapshot-1"),
        run_id="run-1",
        value=1,
    )
    pointer_path = snapshot.pointer_path

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
