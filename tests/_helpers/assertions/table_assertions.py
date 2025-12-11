"""Table-related test assertion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests._helpers.sql import count_nulls, count_table_rows

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def assert_table_has_rows(
    gateway: StorageGateway,
    table: str,
    *,
    min_rows: int = 1,
) -> None:
    """Assert that a table has at least the specified number of rows.

    Parameters
    ----------
    gateway
        Storage gateway with database connection.
    table
        Fully qualified table name (schema.table).
    min_rows
        Minimum number of rows expected.

    Raises
    ------
    AssertionError
        If the table has fewer rows than expected.
    """
    count = count_table_rows(gateway.con, table)
    if count < min_rows:
        message = f"Expected at least {min_rows} rows in {table}, got {count}"
        raise AssertionError(message)


def assert_columns_not_null(
    gateway: StorageGateway,
    table: str,
    columns: list[str],
) -> None:
    """Assert that specified columns contain no NULL values.

    Parameters
    ----------
    gateway
        Storage gateway with database connection.
    table
        Fully qualified table name (schema.table).
    columns
        List of column names to check.

    Raises
    ------
    AssertionError
        If any specified column contains NULL values.
    """
    for col in columns:
        null_count = count_nulls(gateway.con, table, col)
        if null_count > 0:
            message = f"Column {col} in {table} contains {null_count} NULL values"
            raise AssertionError(message)


__all__ = [
    "assert_columns_not_null",
    "assert_table_has_rows",
]
