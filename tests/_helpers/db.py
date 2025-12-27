"""Database helper utilities for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection


def count_rows(
    con: DuckDBPyConnection,
    sql: str,
    params: Sequence[object],
) -> int:
    """Execute a COUNT query and return the integer result.

    Returns
    -------
    int
        Row count for the supplied query.
    """
    result = con.execute(sql, params).fetchone()
    if result is None:
        return 0
    return int(result[0])
