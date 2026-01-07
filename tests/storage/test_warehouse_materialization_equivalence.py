"""Equivalence tests for Warehouse materialization entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.tabular.duckdb_relation import coerce_to_relation
from codeintel.config.primitives import SnapshotRef
from codeintel.core.storage import StorageContext
from codeintel.storage.warehouse import MaterializeOptions, Warehouse
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.columnar_streams import materialize_table_from_rows

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway.protocol import StorageGateway


def test_warehouse_materialize_variants_write_equivalent_rows(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Warehouse materialize_* variants persist equivalent content."""
    warehouse = Warehouse(context=StorageContext(gateway=fresh_gateway))
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    options = MaterializeOptions(snapshot=snapshot, mode="replace")
    table_key = "core.repo_map"

    arrow_table = pa.table({"repo": [snapshot.repo], "commit": [snapshot.commit]})
    relation = coerce_to_relation(fresh_gateway.con, arrow_table, name_hint="repo_map")
    warehouse.materialize_table(table_key, relation, options=options)
    expect_equal(warehouse.count(table_key, snapshot=snapshot), 1, label="df count")

    materialize_table_from_rows(
        warehouse,
        table_key,
        [{"repo": snapshot.repo, "commit": snapshot.commit}],
        columns=("repo", "commit"),
        options=options,
    )
    expect_equal(warehouse.count(table_key, snapshot=snapshot), 1, label="rows count")

    warehouse.materialize_mappings(
        table_key,
        [{"repo": snapshot.repo, "commit": snapshot.commit}],
        columns=("repo", "commit"),
        options=options,
    )
    expect_equal(warehouse.count(table_key, snapshot=snapshot), 1, label="mappings count")

    fetched = fresh_gateway.con.execute(
        "SELECT repo, commit FROM core.repo_map WHERE repo = ? AND commit = ?",
        [snapshot.repo, snapshot.commit],
    ).fetchall()
    expect_equal(len(fetched), 1, label="row count")
    expect_equal(str(fetched[0][0]), snapshot.repo, label="repo")
    expect_equal(str(fetched[0][1]), snapshot.commit, label="commit")
