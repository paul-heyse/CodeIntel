"""Transaction safety tests for snapshot-scoped materialization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.warehouse import MaterializeOptions, Warehouse
from tests._helpers.assertions.expectation_assertions import expect_equal
from tests._helpers.columnar_tables import materialize_table_from_rows

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway.protocol import StorageGateway


def test_materialize_rows_replace_rolls_back_on_write_failure(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """Ensure snapshot-scoped replace is atomic (delete + write)."""
    warehouse = Warehouse(fresh_gateway)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    options = MaterializeOptions(snapshot=snapshot, mode="replace")

    created_at = datetime.now(tz=UTC)
    table_key = "core.repo_map"
    columns = ("repo", "commit", "modules", "overlays", "generated_at")

    materialize_table_from_rows(
        warehouse,
        table_key,
        [(snapshot.repo, snapshot.commit, "[]", None, created_at)],
        columns=columns,
        options=options,
    )

    original = fresh_gateway.con.execute(
        (
            "SELECT repo, commit, CAST(modules AS VARCHAR), overlays, generated_at "
            "FROM core.repo_map WHERE repo = ? AND commit = ?"
        ),
        [snapshot.repo, snapshot.commit],
    ).fetchone()
    if original is None:
        pytest.fail("Expected baseline core.repo_map row to exist")

    bad_rows = [
        (snapshot.repo, snapshot.commit, '["new"]', None, created_at),
        (snapshot.repo, snapshot.commit, '["dupe"]', None, created_at),
    ]
    with pytest.raises(DuckDBError):
        materialize_table_from_rows(
            warehouse,
            table_key,
            bad_rows,
            columns=columns,
            options=options,
        )

    after = fresh_gateway.con.execute(
        (
            "SELECT repo, commit, CAST(modules AS VARCHAR), overlays, generated_at "
            "FROM core.repo_map WHERE repo = ? AND commit = ?"
        ),
        [snapshot.repo, snapshot.commit],
    ).fetchone()
    if after is None:
        pytest.fail("Expected row to remain after failed replace")

    expect_equal(after[0], original[0])
    expect_equal(after[1], original[1])
    expect_equal(after[2], original[2])
    expect_equal(after[3], original[3])
