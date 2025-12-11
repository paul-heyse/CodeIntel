"""SQL-related helpers for test assertions.

These helpers validate identifiers and provide safe, minimal query helpers
for DuckDB-based tests. Prefer using relation methods over raw SQL strings.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from codeintel.storage.gateway.protocol import DuckDBConnection, StorageGateway

if TYPE_CHECKING:
    from collections.abc import Sequence

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
GatewayLike = StorageGateway | DuckDBConnection


def validate_identifier(name: str, *, kind: str = "identifier") -> str:
    """Validate that a string is a safe SQL identifier.

    Parameters
    ----------
    name
        Identifier to validate (schema.table or column name).
    kind
        Description for error messages.

    Returns
    -------
    str
        The original identifier if valid.

    Raises
    ------
    ValueError
        If the identifier contains invalid characters.
    """
    if not _IDENTIFIER_RE.fullmatch(name):
        msg = f"Invalid {kind} identifier: {name!r}"
        raise ValueError(msg)
    return name


def count_table_rows(con: DuckDBConnection, table: str) -> int:
    """Count rows in a table using relation API.

    Parameters
    ----------
    con
        DuckDB connection.
    table
        Fully qualified table name.

    Returns
    -------
    int
        Number of rows in the table.
    """
    safe_table = validate_identifier(table, kind="table")
    relation = con.table(safe_table)
    result = relation.aggregate("count(*)").fetchone()
    return int(result[0]) if result else 0


def count_nulls(con: DuckDBConnection, table: str, column: str) -> int:
    """Count NULLs in a column using relation API.

    Parameters
    ----------
    con
        DuckDB connection.
    table
        Fully qualified table name.
    column
        Column name to inspect.

    Returns
    -------
    int
        Number of NULL entries in the column.
    """
    safe_table = validate_identifier(table, kind="table")
    safe_col = validate_identifier(column, kind="column")
    relation = con.table(safe_table)
    result = relation.filter(f"{safe_col} IS NULL").aggregate("count(*)").fetchone()
    return int(result[0]) if result else 0


def run_query(
    gateway: GatewayLike,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[tuple[object, ...]]:
    """Execute SQL and return all rows as tuples.

    Parameters
    ----------
    gateway
        StorageGateway or DuckDB connection.
    sql
        SQL statement to execute.
    params
        Optional parameter bindings.

    Returns
    -------
    list[tuple[object, ...]]
        Query results as tuples.
    """
    con = gateway if isinstance(gateway, DuckDBConnection) else gateway.con
    return [tuple(row) for row in con.execute(sql, params).fetchall()]


def expect_single_value(row: Sequence[object] | None, *, message: str | None = None) -> object:
    """Extract a single value from a one-column row.

    Parameters
    ----------
    row
        Row returned from ``fetchone()`` (tuple or list).
    message
        Optional override for the assertion message.

    Returns
    -------
    object
        The single value contained in the row.

    Raises
    ------
    AssertionError
        If the row is missing or does not contain exactly one value.
    """
    if row is None:
        raise AssertionError(message or "Expected one row but query returned none")
    if len(row) != 1:
        raise AssertionError(message or f"Expected one column, found {len(row)}")
    return row[0]


__all__ = [
    "count_nulls",
    "count_table_rows",
    "expect_single_value",
    "run_query",
    "validate_identifier",
]
