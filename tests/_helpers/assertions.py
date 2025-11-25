"""Assertion helpers for seeded DuckDB gateways in tests."""

from __future__ import annotations

import re
from collections.abc import Iterable

from codeintel.storage.gateway import StorageGateway

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")


def assert_table_has_rows(gateway: StorageGateway, table: str, *, min_rows: int = 1) -> None:
    """
    Ensure a table contains at least `min_rows` rows.

    Raises
    ------
    AssertionError
        If the table has fewer rows than expected.
    """
    if not _IDENTIFIER_PATTERN.fullmatch(table):
        message = f"Unsafe table identifier: {table}"
        raise AssertionError(message)
    count_row = gateway.con.table(table).aggregate("count(*)").fetchone()
    count = int(count_row[0]) if count_row is not None else 0
    if count < min_rows:
        message = f"Expected at least {min_rows} rows in {table}, found {count}"
        raise AssertionError(message)


def assert_columns_not_null(
    gateway: StorageGateway,
    table: str,
    columns: Iterable[str],
) -> None:
    """
    Assert that the specified columns do not contain NULL values.

    Raises
    ------
    AssertionError
        If any column contains NULL values.
    """
    if not _IDENTIFIER_PATTERN.fullmatch(table):
        message = f"Unsafe table identifier: {table}"
        raise AssertionError(message)
    for column in columns:
        if not _IDENTIFIER_PATTERN.fullmatch(column):
            message = f"Unsafe column identifier: {column}"
            raise AssertionError(message)
        rel = gateway.con.table(table).filter(f"{column} IS NULL")
        row = rel.aggregate("count(*)").fetchone()
        null_count = int(row[0]) if row is not None else 0
        if null_count > 0:
            message = f"Column {column} in {table} contains {null_count} NULL rows"
            raise AssertionError(message)
