"""Tests for Hamilton materializer utilities.

These tests validate the Hamilton-native DataSaver implementations used for
DAG-visible I/O, replacing the legacy ``native.materializer`` utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from codeintel.build.contracts import OutputContract
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver, DuckDBRowsSaver
from codeintel.build.schemas.column_resolution import deferred_columns_for_table_key
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)
from tests._helpers.build import make_build_config, make_build_paths
from tests._helpers.fakes.fake_providers import FakeProviders

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.providers import Providers
    from codeintel.storage.gateway import StorageGateway


def _modules_rows(*, repo: str, commit: str, count: int) -> pd.DataFrame:
    rows = [
        {
            "module": f"m{idx}",
            "path": f"pkg/mod_{idx}.py",
            "repo": repo,
            "commit": commit,
            "language": "python",
            "tags": "[]",
            "owners": "[]",
        }
        for idx in range(count)
    ]
    return pd.DataFrame(rows)


def _make_env(*, gateway: StorageGateway, snapshot: SnapshotRef) -> BuildEnv:
    """Create a minimal BuildEnv suitable for saver execution in tests.

    Returns
    -------
    BuildEnv
        Build environment for saver execution.
    """
    tmp_path = snapshot.repo_root
    paths = make_build_paths(tmp_path)
    config = make_build_config()
    providers = cast("Providers", FakeProviders.defaults())
    return BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        force_targets=frozenset({"modules"}),
    )


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
            contract=OutputContract.simple(table_keys=("core.modules",)),
        )
    )
    return graph


def test_materialize_table_uses_policy_and_insert_select(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """DuckDBIbisTableSaver should replace snapshot rows via Warehouse policy."""
    repo = "r"
    commit = "c"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()
    saver = DuckDBIbisTableSaver(
        env=env,
        graph=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df1 = _modules_rows(repo=repo, commit=commit, count=1)
    fresh_gateway.con.register("tmp_modules_1", df1)
    expr1 = fresh_gateway.ibis.con.table("tmp_modules_1")
    meta1 = saver.save_data(expr1)
    expect_equal(meta1["status"], expected="succeeded")
    expect_equal(meta1["row_count"], expected=1)

    df2 = _modules_rows(repo=repo, commit=commit, count=2)
    fresh_gateway.con.register("tmp_modules_2", df2)
    expr2 = fresh_gateway.ibis.con.table("tmp_modules_2")
    meta2 = saver.save_data(expr2)
    expect_equal(meta2["status"], expected="succeeded")
    expect_equal(meta2["row_count"], expected=2)

    row = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row is not None, message="Expected COUNT(*) query to return a row")
    row_tuple = cast("tuple[int, ...]", row)
    expect_equal(row_tuple[0], expected=2)


def test_materialize_table_validates_when_schema_available(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """DuckDBIbisTableSaver should succeed when schema validation is enabled."""
    repo = "r"
    commit = "c"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    env = BuildEnv(
        gateway=env.gateway,
        snapshot=env.snapshot,
        paths=env.paths,
        providers=env.providers,
        config=env.config,
        force_targets=env.force_targets,
        manifest_index=env.manifest_index,
        validate_outputs=True,
        strict_contracts=env.strict_contracts,
        wrapper_allowlist=env.wrapper_allowlist,
        fingerprint_policy=env.fingerprint_policy,
    )
    graph = _make_graph()
    saver = DuckDBIbisTableSaver(
        env=env,
        graph=graph,
        target_name="modules",
        table_key="core.modules",
    )

    df = _modules_rows(repo=repo, commit=commit, count=2)
    fresh_gateway.con.register("tmp_modules_validate", df)
    expr = fresh_gateway.ibis.con.table("tmp_modules_validate")
    meta = saver.save_data(expr)
    expect_equal(meta["status"], expected="succeeded")
    expect_equal(meta["row_count"], expected=len(df))


def test_rows_saver_resolves_deferred_columns(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """DuckDBRowsSaver should resolve deferred columns at execution time."""
    repo = "r"
    commit = "c"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    env = _make_env(gateway=fresh_gateway, snapshot=snapshot)
    graph = _make_graph()
    table_key = "core.modules"
    target = graph.get("modules")

    schema = get_schema_service().require_table_schema(table_key)
    column_names = tuple(col.name for col in schema.columns)
    values_by_column: dict[str, object] = {}
    for column in schema.columns:
        if column.type == "JSON":
            values_by_column[column.name] = "[]"
        elif column.type in {"INTEGER", "BIGINT", "DECIMAL(38,0)"}:
            values_by_column[column.name] = 1
        elif column.type == "DOUBLE":
            values_by_column[column.name] = 1.0
        elif column.type == "BOOLEAN":
            values_by_column[column.name] = True
        else:
            values_by_column[column.name] = f"value_{column.name}"
    values_by_column["repo"] = repo
    values_by_column["commit"] = commit
    row = tuple(values_by_column[name] for name in column_names)

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

    row_result = fresh_gateway.con.execute(
        "SELECT * FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row_result is not None, message="Expected row materialization to persist data")
    persisted = cast("tuple[object, ...]", row_result)
    expect_equal(persisted, expected=row)
