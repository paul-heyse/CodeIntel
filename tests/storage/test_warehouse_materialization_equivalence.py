"""Equivalence tests for Warehouse materialization entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.warehouse import MaterializeOptions, Warehouse
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway.protocol import StorageGateway


def test_warehouse_materialize_variants_write_equivalent_rows(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    warehouse = Warehouse(fresh_gateway)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    options = MaterializeOptions(snapshot=snapshot, mode="replace")
    table_key = "core.repo_map"

    dataframe = pd.DataFrame([{"repo": snapshot.repo, "commit": snapshot.commit}])
    warehouse.materialize_dataframe(table_key, dataframe, options=options)
    expect_equal(warehouse.count(table_key, snapshot=snapshot), 1, label="df count")

    warehouse.materialize_rows(
        table_key,
        [(snapshot.repo, snapshot.commit)],
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

