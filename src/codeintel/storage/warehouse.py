"""Typed warehouse API for storage.

This module defines a small, typed surface area intended to be the single I/O
boundary for build + serving over time. It is implemented as a thin wrapper
around `StorageGateway` and existing policy/ibis primitives.

The API is intentionally conservative to avoid forcing immediate refactors; it
can be adopted incrementally by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from duckdb import ColumnExpression, ConstantExpression

from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.ibis_types import filter_by

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBRelation

from codeintel.storage.gateway.protocol import DuckDBError

WriteMode = Literal["append", "replace"]


@dataclass(frozen=True, slots=True)
class Warehouse:
    """Warehouse façade over `StorageGateway`.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB + Ibis access.
    """

    gateway: StorageGateway

    def read(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> ir.Table:
        """Return an Ibis table expression, optionally snapshot-filtered.

        Snapshot filtering is applied only when both `repo` and `commit` columns
        exist on the table and a snapshot is provided.

        Returns
        -------
        ir.Table
            Ibis expression for the requested table, optionally filtered.
        """
        expr = self.gateway.ibis.table(table_key)
        if snapshot is None:
            return expr

        schema = expr.schema()
        names = set(schema.keys())
        if "repo" in names and "commit" in names:
            return filter_by(expr, expr["repo"] == snapshot.repo, expr["commit"] == snapshot.commit)
        return expr

    def exists(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> bool:
        """Return True if the table/view exists.

        When `snapshot` is provided, this also checks for the presence of at
        least one row matching `repo` and `commit`.

        Returns
        -------
        bool
            True when the object exists (and has snapshot rows when requested).
        """
        schema, name = split_table_key(table_key)
        try:
            relation = self.gateway.con.table(f"{schema}.{name}")
        except DuckDBError:
            return False

        if snapshot is None:
            return True

        return _relation_has_snapshot_rows(
            relation,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )

    def count(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> int:
        """Count rows in a table, optionally snapshot-filtered.

        Returns
        -------
        int
            Row count for the requested object.
        """
        schema, name = split_table_key(table_key)
        relation = self.gateway.con.table(f"{schema}.{name}")
        if snapshot is not None:
            relation = relation.filter(
                (ColumnExpression("repo") == ConstantExpression(snapshot.repo))
                & (ColumnExpression("commit") == ConstantExpression(snapshot.commit))
            )
        row = relation.count("*").fetchone()
        return int(row[0]) if row is not None else 0


def _relation_has_snapshot_rows(relation: DuckDBRelation, *, repo: str, commit: str) -> bool:
    try:
        filtered = relation.filter(
            (ColumnExpression("repo") == ConstantExpression(repo))
            & (ColumnExpression("commit") == ConstantExpression(commit))
        )
        return filtered.limit(1).fetchone() is not None
    except DuckDBError:
        return False


__all__ = ["Warehouse", "WriteMode"]
