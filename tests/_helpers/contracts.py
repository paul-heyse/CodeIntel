"""Shared contract and assertion helpers for analytics tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection

    from codeintel.storage.gateway import StorageGateway


@dataclass
class ContractCtx:
    """Concrete contract context for contract checkers."""

    gateway: StorageGateway
    repo: str
    commit: str


def count_rows(con: DuckDBPyConnection, sql: str, params: Sequence[object]) -> int:
    """
    Return integer row count for a parameterized query.

    Parameters
    ----------
    con
        DuckDB connection used to execute the query.
    sql
        Parameterized SQL statement to run.
    params
        Parameters bound to the query.

    Returns
    -------
    int
        Row count for the query (0 when no rows are returned).
    """
    row = con.execute(sql, params).fetchone()
    if row is None:
        return 0
    return int(row[0])


__all__ = [
    "ContractCtx",
    "count_rows",
]
