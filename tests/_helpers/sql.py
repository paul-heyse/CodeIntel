"""SQL-related helpers for test assertions.

These helpers validate identifiers and provide safe, minimal query helpers
for DuckDB-based tests. Prefer using relation methods over raw SQL strings.
"""

from __future__ import annotations

import re

from codeintel.storage.gateway.protocol import DuckDBConnection

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


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


__all__ = [
    "count_nulls",
    "count_table_rows",
    "validate_identifier",
]
