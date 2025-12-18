"""Tests for the DuckDB read-only pool."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import duckdb
import pytest

from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from pathlib import Path

POOL_SIZE = 3


def _make_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE kv (k INTEGER, v VARCHAR)")
    con.execute("INSERT INTO kv VALUES (1, 'one'), (2, 'two')")
    con.close()


def test_pool_creates_configured_size(tmp_path: Path) -> None:
    """Pool initializes N connections."""
    db_path = tmp_path / "db.duckdb"
    _make_db(db_path)

    pool = ReadPoolWarehouse(db_path, PoolConfig(size=POOL_SIZE))
    try:
        with ExitStack() as stack:
            warehouses = [stack.enter_context(pool.acquire()) for _ in range(POOL_SIZE)]
            conn_ids = {id(warehouse.gateway.con) for warehouse in warehouses}
            expect_equal(len(conn_ids), POOL_SIZE)
    finally:
        pool.close_gracefully()


def test_pool_acquire_release_cycle(tmp_path: Path) -> None:
    """Pool reuses connections across acquire/release cycles."""
    db_path = tmp_path / "db.duckdb"
    _make_db(db_path)

    pool = ReadPoolWarehouse(db_path, PoolConfig(size=1))
    with pool.acquire() as warehouse1:
        con1 = warehouse1.gateway.con
        expect_equal(con1.execute("SELECT COUNT(*) FROM kv").fetchone(), (2,))

    with pool.acquire() as warehouse2:
        con2 = warehouse2.gateway.con
        expect_equal(con2.execute("SELECT COUNT(*) FROM kv").fetchone(), (2,))
        expect_equal(id(con1), id(con2))
    pool.close_gracefully()


def test_pool_close_gracefully_closes_available(tmp_path: Path) -> None:
    """Graceful close drains available connections and prevents new acquires."""
    db_path = tmp_path / "db.duckdb"
    _make_db(db_path)

    pool = ReadPoolWarehouse(db_path, PoolConfig(size=1))
    pool.close_gracefully()

    with pytest.raises(RuntimeError, match="Pool is closing"), pool.acquire():
        pass
