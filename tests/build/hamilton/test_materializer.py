"""Tests for Hamilton materializer utilities.

These tests validate the Hamilton-native DataSaver implementations used for
DAG-visible I/O, replacing the legacy ``native.materializer`` utilities.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pandas as pd

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver, DuckDBRowsSaver
from codeintel.build.schemas.column_resolution import deferred_columns_for_table_key
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.core.hashing import stable_hash
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)
from tests._helpers.contracts import contract_for_keys
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import Column


def _modules_rows(*, repo: str, commit: str, count: int) -> pd.DataFrame:
    rows = []
    for idx in range(count):
        row = {
            "module": f"m{idx}",
            "path": f"pkg/mod_{idx}.py",
            "repo": repo,
            "commit": commit,
            "language": "python",
            "tags": "[]",
            "owners": "[]",
        }
        row["row_hash"] = stable_hash(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_graph() -> TargetGraph:
    """Create a minimal TargetGraph that contains a modules target.

    Returns
    -------
    TargetGraph
        Target graph containing only the modules target.
    """
    graph = TargetGraph()
    graph.register(
        OutputTarget(
            name="modules",
            module="ingestion",
            contract=contract_for_keys(("core.modules",)),
        )
    )
    return graph


def _module_row_for_schema(
    *,
    repo: str,
    commit: str,
    schema_columns: tuple[Column, ...],
) -> tuple[object, ...]:
    """Build a row tuple matching the schema column ordering.

    Returns
    -------
    tuple[object, ...]
        Row tuple matching schema column ordering.
    """
    column_names = tuple(column.name for column in schema_columns)
    values_by_column: dict[str, object] = {}
    for column in schema_columns:
        col_name = column.name
        col_type = column.type
        if col_type == "JSON":
            values_by_column[col_name] = "[]"
        elif col_type in {"INTEGER", "BIGINT", "DECIMAL(38,0)"}:
            values_by_column[col_name] = 1
        elif col_type == "DOUBLE":
            values_by_column[col_name] = 1.0
        elif col_type == "BOOLEAN":
            values_by_column[col_name] = True
        else:
            values_by_column[col_name] = f"value_{col_name}"
    values_by_column["repo"] = repo
    values_by_column["commit"] = commit
    return tuple(values_by_column[name] for name in column_names)


def test_materialize_table_uses_policy_and_insert_select(
    build_harness: HamiltonBuildHarness,
) -> None:
    """DuckDBIbisTableSaver should replace snapshot rows via Warehouse policy."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    snapshot = env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    graph = _make_graph()
    saver = DuckDBIbisTableSaver(
        env=env,
        graph=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df1 = _modules_rows(repo=repo, commit=commit, count=1)
    env.gateway.con.register("tmp_modules_1", df1)
    expr1 = env.gateway.ibis.con.table("tmp_modules_1")
    meta1 = saver.save_data(expr1)
    expect_equal(meta1["status"], expected="succeeded")
    expect_equal(meta1["row_count"], expected=1)

    df2 = _modules_rows(repo=repo, commit=commit, count=2)
    env.gateway.con.register("tmp_modules_2", df2)
    expr2 = env.gateway.ibis.con.table("tmp_modules_2")
    meta2 = saver.save_data(expr2)
    expect_equal(meta2["status"], expected="succeeded")
    expect_equal(meta2["row_count"], expected=2)

    row = env.gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row is not None, message="Expected COUNT(*) query to return a row")
    row_tuple = cast("tuple[int, ...]", row)
    expect_equal(row_tuple[0], expected=2)


def test_materialize_table_validates_when_schema_available(
    build_harness: HamiltonBuildHarness,
) -> None:
    """DuckDBIbisTableSaver should succeed when schema validation is enabled."""
    harness = build_harness.with_force_targets("modules")
    env = replace(harness.build_env(), validate_outputs=True)
    repo = env.snapshot.repo
    commit = env.snapshot.commit
    graph = _make_graph()
    saver = DuckDBIbisTableSaver(
        env=env,
        graph=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=repo, commit=commit, count=2)
    env.gateway.con.register("tmp_modules_validate", df)
    expr = env.gateway.ibis.con.table("tmp_modules_validate")
    meta = saver.save_data(expr)
    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=len(df))


def test_rows_saver_resolves_deferred_columns(
    build_harness: HamiltonBuildHarness,
) -> None:
    """DuckDBRowsSaver should resolve deferred columns at execution time."""
    harness = build_harness.with_force_targets("modules")
    env = harness.build_env()
    snapshot = env.snapshot
    repo = snapshot.repo
    commit = snapshot.commit
    graph = _make_graph()
    table_key = "core.modules"
    target = graph.get("modules")

    schema = get_schema_service().require_table_schema(table_key)
    row = _module_row_for_schema(
        repo=repo,
        commit=commit,
        schema_columns=tuple(schema.columns),
    )

    saver = DuckDBRowsSaver(
        env=env,
        graph=graph,
        target_name="modules",
        table_key=table_key,
        columns=deferred_columns_for_table_key(table_key),
    )

    with ContractEnforcer.for_target(target, strict=True):
        meta = saver.save_data((row,))

    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=1)

    row_result = env.gateway.con.execute(
        "SELECT * FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row_result is not None, message="Expected row materialization to persist data")
    persisted = cast("tuple[object, ...]", row_result)
    expect_equal(persisted, expected=row)
