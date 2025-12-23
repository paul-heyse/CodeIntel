"""Compiler upgrade gates for serving semantic queries.

These tests intentionally snapshot the canonicalized SQL produced by the Ibis
compiler for representative semantic query shapes. They serve as an early
warning system for SQLGlot/Ibis upgrades that change SQL rendering in ways that
may affect caching, query policies, or downstream tooling.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest
from sqlglot import parse_one

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import FilterSpec, SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts

if TYPE_CHECKING:
    from pathlib import Path


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
    return parse_one(sql, dialect=DUCKDB_DIALECT).sql(dialect=DUCKDB_DIALECT)


def _make_snapshot_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA docs")
    con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    con.close()


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
    payload = {
        "db_path": str(db_path),
        "semantic_registry_path": str(registry_path),
        "schema_manifest_path": str(manifest_path),
        "buildspec_path": str(buildspec_path),
        "repo": "demo/repo",
        "commit": "deadbeef",
        "run_id": "run-1",
        "published_at": datetime.now(tz=UTC).isoformat(),
        "semantic_layer_version": "v123",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.anyio
async def test_compiler_upgrade_gate_numeric_filter_sql_is_stable(tmp_path: Path) -> None:
    """Canonical SQL for a numeric filter + order + limit remains stable."""
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
        'SELECT * FROM "docs"."v_demo" AS "t0" WHERE "t0"."id" >= 2 '
        'ORDER BY "t0"."id" DESC LIMIT 10'
    )
    expect_equal(canonical, expected)


@pytest.mark.anyio
async def test_compiler_upgrade_gate_string_contains_sql_is_stable(tmp_path: Path) -> None:
    """Canonical SQL for a string predicate remains stable."""
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
        'SELECT * FROM "docs"."v_demo" AS "t0" WHERE CONTAINS("t0"."label", \'t\') '
        'ORDER BY "t0"."id" ASC LIMIT 10'
    )
    expect_equal(canonical, expected)
