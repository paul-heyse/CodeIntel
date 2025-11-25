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


def expect_true(condition: object, message: str | None = None) -> None:
    """
    Raise AssertionError when a boolean condition is not satisfied.

    Parameters
    ----------
    condition : object
        Value that must evaluate to True.
    message : str | None
        Optional override for the assertion message.

    Raises
    ------
    AssertionError
        If the condition evaluates to False.
    """
    if not condition:
        raise AssertionError(message or "Expected condition to be True.")


def expect_equal(actual: object, expected: object, message: str | None = None) -> None:
    """
    Raise AssertionError when two values differ.

    Parameters
    ----------
    actual : object
        Observed value.
    expected : object
        Expected value to compare against.
    message : str | None
        Optional override for the assertion message.

    Raises
    ------
    AssertionError
        If the values are not equal.
    """
    if actual != expected:
        detail = message or f"Expected {expected!r}, got {actual!r}"
        raise AssertionError(detail)


def expect_in(value: object, container: Iterable[object], message: str | None = None) -> None:
    """
    Raise AssertionError when a value is not present in a container.

    Parameters
    ----------
    value : object
        Item expected to be present.
    container : Iterable[object]
        Container to inspect for membership.
    message : str | None
        Optional override for the assertion message.

    Raises
    ------
    AssertionError
        If the value is missing from the container.
    """
    if value not in container:
        detail = message or f"Expected {value!r} to be in {container!r}"
        raise AssertionError(detail)
