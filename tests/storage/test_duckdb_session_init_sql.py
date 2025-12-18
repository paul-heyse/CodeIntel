"""Tests that session/open paths apply init SQL consistently."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.factory import open_gateway
from codeintel.storage.gateway.pool import PoolConfig, ReadPoolWarehouse
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_is_not_none


def test_open_gateway_applies_init_sql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEINTEL_DUCKDB_INIT_SQL", "CREATE TEMP TABLE ci_init_test(x INTEGER);")

    db_path = tmp_path / "session_init.duckdb"
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    gateway = open_gateway(cfg)
    try:
        row = gateway.con.execute("SELECT COUNT(*) FROM ci_init_test").fetchone()
        expect_is_not_none(row, label="ci_init_test count row")
        expect_equal(int(row[0]), 0, label="ci_init_test count")
    finally:
        gateway.close()


def test_read_pool_applies_init_sql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEINTEL_DUCKDB_INIT_SQL", "CREATE TEMP TABLE ci_init_test2(x INTEGER);")

    db_path = tmp_path / "pool.duckdb"
    con = duckdb.connect(str(db_path))
    con.close()

    pool = ReadPoolWarehouse(db_path, PoolConfig(size=1))
    try:
        with pool.acquire() as warehouse:
            row = warehouse.gateway.con.execute("SELECT COUNT(*) FROM ci_init_test2").fetchone()
            expect_is_not_none(row, label="ci_init_test2 count row")
            expect_equal(int(row[0]), 0, label="ci_init_test2 count")
    finally:
        pool.close_gracefully()
