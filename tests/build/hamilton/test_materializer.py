"""Tests for Hamilton native materializer utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_table
from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)

if TYPE_CHECKING:
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


def test_materialize_table_uses_policy_and_insert_select(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """materialize_table should replace snapshot rows via INSERT...SELECT."""
    repo = "r"
    commit = "c"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    ctx = MaterializationContext(gateway=fresh_gateway, snapshot=snapshot, validate=False)

    df1 = _modules_rows(repo=repo, commit=commit, count=1)
    fresh_gateway.con.register("tmp_modules_1", df1)
    expr1 = fresh_gateway.ibis.con.table("tmp_modules_1")
    ref1 = materialize_table(ctx, "core.modules", expr1)
    expect_equal(ref1.row_count, 1)

    df2 = _modules_rows(repo=repo, commit=commit, count=2)
    fresh_gateway.con.register("tmp_modules_2", df2)
    expr2 = fresh_gateway.ibis.con.table("tmp_modules_2")
    ref2 = materialize_table(ctx, "core.modules", expr2)
    expect_equal(ref2.row_count, 2)

    row = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo=? AND commit=?",
        [repo, commit],
    ).fetchone()
    expect_true(row is not None, message="Expected COUNT(*) query to return a row")
    expect_equal(int(row[0]), 2)


def test_materialize_table_validates_when_schema_available(
    fresh_gateway: StorageGateway,
    tmp_path: Path,
) -> None:
    """materialize_table should validate DataFrame when schema is present."""
    repo = "r"
    commit = "c"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=tmp_path / "repo")
    ctx = MaterializationContext(gateway=fresh_gateway, snapshot=snapshot, validate=True)

    df = _modules_rows(repo=repo, commit=commit, count=2)
    fresh_gateway.con.register("tmp_modules_validate", df)
    expr = fresh_gateway.ibis.con.table("tmp_modules_validate")

    class StubSchema:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def validate(self, frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
            self.calls.append({"frame": frame.copy(), "lazy": lazy})
            return frame

    schema = StubSchema()
    ref = materialize_table(
        ctx,
        "core.modules",
        expr,
        schema_resolver=lambda _: schema,
    )

    expect_true(bool(schema.calls), message="Schema.validate should be invoked")
    expect_equal(schema.calls[0]["lazy"], expected=False)
    expect_equal(ref.row_count, len(df))
