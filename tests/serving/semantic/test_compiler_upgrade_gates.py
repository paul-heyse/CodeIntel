"""Compiler upgrade gates for serving semantic queries.

These tests intentionally snapshot the canonicalized SQL produced by the Ibis
compiler for representative semantic query shapes. They serve as an early
warning system for SQLGlot/Ibis upgrades that change SQL rendering in ways that
may affect caching, query policies, or downstream tooling.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from sqlglot import parse_one

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import FilterSpec, SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.constants import DUCKDB_DIALECT
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


def _canonical_duckdb_sql(sql: str) -> str:
    """Return canonical DuckDB SQL text for stable golden comparisons.

    Parameters
    ----------
    sql
        DuckDB SQL text to normalize.

    Returns
    -------
    str
        Canonicalized SQL string.
    """
    canonical = parse_one(sql, dialect=DUCKDB_DIALECT).sql(dialect=DUCKDB_DIALECT)
    canonical = _PARQUET_SCAN_PATH_RE.sub("PARQUET_SCAN(['<DATASET>']", canonical)
    return _PARQUET_SCAN_ALIAS_RE.sub("parquet_scan", canonical)


_PARQUET_SCAN_PATH_RE = re.compile(r"PARQUET_SCAN\(\['[^']+'\]")
_PARQUET_SCAN_ALIAS_RE = re.compile(r"parquet_[0-9a-f]+")


def _make_snapshot_db(db_path: Path) -> None:
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        ensure_production_schemas(gateway.con)
        gateway.con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
        gateway.con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
        gateway.con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    finally:
        gateway.close()


def _compiler_views() -> list[dict[str, object]]:
    return [
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
    ]


def _compiler_tables() -> list[dict[str, object]]:
    return [
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
    ]


def _make_snapshot(factory: ServingSnapshotFactory) -> ServingSnapshot:
    artifacts = SnapshotArtifacts(
        views=_compiler_views(),
        tables=_compiler_tables(),
        db_setup=_make_snapshot_db,
    )
    return factory.make_snapshot(artifacts=artifacts)


@pytest.mark.anyio
async def test_compiler_upgrade_gate_numeric_filter_sql_is_stable(tmp_path: Path) -> None:
    """Canonical SQL for a numeric filter + order + limit remains stable."""
    snapshot = _make_snapshot(ServingSnapshotFactory(tmp_path))

    manager = ServingDBManager(
        pointer_path=snapshot.pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
        hot_swap=False,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="polars",
                schema_enforcement="strict",
            ),
        )
        request = SemanticQueryRequest(
            view_id="demo.view",
            filters=[FilterSpec(column="id", op="gte", value=2)],
            order_by=["-id"],
            limit=10,
            offset=0,
        )
        explain = kernel.explain(request)
        canonical = _canonical_duckdb_sql(explain.sql)
    finally:
        await manager.stop()

    expected = (
        'SELECT id, "label" FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM '
        "PARQUET_SCAN(['<DATASET>'], (binary_as_string = FALSE), (file_row_number = FALSE), "
        "(filename = FALSE), (hive_partitioning = FALSE), (union_by_name = TRUE))) AS "
        "parquet_scan WHERE (id >= 2)) AS parquet_scan ORDER BY id DESC) AS parquet_scan LIMIT 10"
    )
    expect_equal(canonical, expected)


@pytest.mark.anyio
async def test_compiler_upgrade_gate_string_contains_sql_is_stable(tmp_path: Path) -> None:
    """Canonical SQL for a string predicate remains stable."""
    snapshot = _make_snapshot(ServingSnapshotFactory(tmp_path))

    manager = ServingDBManager(
        pointer_path=snapshot.pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
        hot_swap=False,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="polars",
                schema_enforcement="strict",
            ),
        )
        request = SemanticQueryRequest(
            view_id="demo.view",
            filters=[FilterSpec(column="label", op="contains", value="t")],
            order_by=["id"],
            limit=10,
            offset=0,
        )
        explain = kernel.explain(request)
        canonical = _canonical_duckdb_sql(explain.sql)
    finally:
        await manager.stop()

    expected = (
        'SELECT id, "label" FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM '
        "PARQUET_SCAN(['<DATASET>'], (binary_as_string = FALSE), (file_row_number = FALSE), "
        "(filename = FALSE), (hive_partitioning = FALSE), (union_by_name = TRUE))) AS "
        "parquet_scan WHERE CONTAINS(\"label\", 't')) AS parquet_scan ORDER BY id) AS "
        "parquet_scan LIMIT 10"
    )
    expect_equal(canonical, expected)
