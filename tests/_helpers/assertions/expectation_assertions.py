"""Expectation-style assertion helpers for test validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import duckdb


def _prefix(label: str | None) -> str:
    """Build optional label prefix for failure messages.

    Returns
    -------
    str
        Label with colon/space suffix, or empty string if no label.
    """
    return f"{label}: " if label else ""


def expect_true(condition: object, *, message: str | None = None) -> None:
    """Raise AssertionError when a boolean condition is not satisfied.

    Parameters
    ----------
    condition
        Value that must evaluate to True.
    message
        Optional override for the assertion message.

    Raises
    ------
    AssertionError
        If the condition evaluates to False.
    """
    if not condition:
        raise AssertionError(message or "Expected condition to be True.")


def expect_false(condition: object, *, message: str | None = None) -> None:
    """Raise AssertionError when a boolean condition is unexpectedly True.

    Raises
    ------
    AssertionError
        If the condition evaluates to True.
    """
    if condition:
        raise AssertionError(message or "Expected condition to be False.")


def expect_equal(actual: object, expected: object, *, label: str | None = None) -> None:
    """Raise AssertionError when two values differ.

    Parameters
    ----------
    actual
        Observed value.
    expected
        Expected value to compare against.
    label
        Optional label prefix for the failure message.

    Raises
    ------
    AssertionError
        If the values are not equal.
    """
    if actual != expected:
        failure_message = f"{_prefix(label)}expected {expected!r}, got {actual!r}"
        raise AssertionError(failure_message)


def expect_not_equal(actual: object, expected: object, *, label: str | None = None) -> None:
    """Raise AssertionError when two values are unexpectedly equal.

    Raises
    ------
    AssertionError
        If the values are equal.
    """
    if actual == expected:
        failure_message = f"{_prefix(label)}did not expect {expected!r}"
        raise AssertionError(failure_message)


def expect_in(value: object, container: Iterable[object], *, label: str | None = None) -> None:
    """Raise AssertionError when a value is not present in a container.

    Parameters
    ----------
    value
        Item expected to be present.
    container
        Container to inspect for membership.
    label
        Optional label prefix for the failure message.

    Raises
    ------
    AssertionError
        If the value is missing from the container.
    """
    if value in container:
        return
    failure_message = f"{_prefix(label)}{value!r} not found in {container!r}"
    raise AssertionError(failure_message)


def expect_not_in(
    value: object,
    container: Iterable[object],
    *,
    label: str | None = None,
) -> None:
    """Raise AssertionError when a value is present in a container.

    Parameters
    ----------
    value
        Item expected to be absent.
    container
        Container to inspect for membership.
    label
        Optional label prefix for the failure message.

    Raises
    ------
    AssertionError
        If the value is present in the container.
    """
    if value not in container:
        return
    failure_message = f"{_prefix(label)}{value!r} unexpectedly found in {container!r}"
    raise AssertionError(failure_message)


def expect_row_count(rows: Sequence[object], expected: int, *, label: str | None = None) -> None:
    """Raise AssertionError when a row sequence length differs from expected.

    Parameters
    ----------
    rows
        Sequence of row-like objects.
    expected
        Expected number of rows.
    label
        Optional label prefix for the failure message.

    Raises
    ------
    AssertionError
        If the sequence length does not match ``expected``.
    """
    actual = len(rows)
    if actual != expected:
        failure_message = f"{_prefix(label)}expected {expected} rows, got {actual}"
        raise AssertionError(failure_message)


def expect_table_row_count(
    con: duckdb.DuckDBPyConnection,
    table: str,
    expected_count: int,
    *,
    label: str | None = None,
) -> None:
    """Assert that a table has the expected row count."""
    row = con.table(table).aggregate("count(*)").fetchone()
    actual = int(row[0]) if row is not None else 0
    expect_equal(actual, expected_count, label=label or f"{table}_row_count")


def expect_table_schema(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    expected_columns: Mapping[str, str],
    label: str | None = None,
) -> None:
    """Assert that a table's schema matches expected column names/types."""
    rows = con.execute("PRAGMA table_info(?)", [table]).fetchall()
    observed = {cast("str", row[1]): cast("str", row[2]) for row in rows} if rows else {}
    expect_equal(observed, dict(expected_columns), label=label or f"{table}_schema")


def expect_is_not(actual: object, unexpected: object, *, label: str | None = None) -> None:
    """
    Raise AssertionError when two references are unexpectedly identical.

    Raises
    ------
    AssertionError
        If `actual` is the same object as `unexpected`.
    """
    if actual is unexpected:
        failure_message = f"{_prefix(label)}did not expect {unexpected!r} identity match"
        raise AssertionError(failure_message)


def expect_is_instance(
    value: object,
    expected_type: type[object],
    *,
    label: str | None = None,
) -> None:
    """Assert that a value is an instance of a type.

    Parameters
    ----------
    value
        Object to check.
    expected_type
        Type that value should be an instance of.
    label
        Optional label prefix for the failure message.

    Raises
    ------
    AssertionError
        If ``value`` is not an instance of ``expected_type``.
    """
    if isinstance(value, expected_type):
        return
    failure_message = f"{_prefix(label)}expected instance of {expected_type!r}, got {type(value)!r}"
    raise AssertionError(failure_message)


def expect_is_none(value: object, *, label: str | None = None) -> None:
    """Raise AssertionError when value is not None.

    Raises
    ------
    AssertionError
        If ``value`` is not ``None``.
    """
    if value is not None:
        failure_message = f"{_prefix(label)}expected None, got {value!r}"
        raise AssertionError(failure_message)


def expect_is_not_none[T](
    value: T | None, *, label: str | None = None, message: str | None = None
) -> T:
    """Raise AssertionError when value is None and return the unwrapped value.

    Returns
    -------
    T
        The provided ``value`` when it is not ``None``.

    Raises
    ------
    AssertionError
        If ``value`` is ``None``.
    """
    if value is None:
        failure_message = message or f"{_prefix(label)}unexpected None value"
        raise AssertionError(failure_message)
    return value


def expect_length(
    sequence: Iterable[object],
    expected: int,
    *,
    label: str | None = None,
) -> None:
    """Assert that a sequence has the expected length.

    Parameters
    ----------
    sequence
        Iterable to measure.
    expected
        Expected length.
    label
        Optional label prefix for the failure message.

    Raises
    ------
    AssertionError
        If the length differs from ``expected``.
    """
    actual = len(list(sequence))
    if actual == expected:
        return
    failure_message = f"{_prefix(label)}expected length {expected}, got {actual}"
    raise AssertionError(failure_message)


def expect_empty(sequence: Iterable[object], *, label: str | None = None) -> None:
    """Assert that a sequence is empty."""
    expect_length(sequence, 0, label=label)


def expect_not_empty(sequence: Iterable[object], *, label: str | None = None) -> None:
    """Assert that a sequence is not empty.

    Raises
    ------
    AssertionError
        If the sequence is empty.
    """
    if any(True for _ in sequence):
        return
    failure_message = f"{_prefix(label)}expected non-empty sequence"
    raise AssertionError(failure_message)


def unwrap_optional[T](
    value: T | None, *, message: str | None = None, label: str | None = None
) -> T:
    """Return a non-optional value or raise AssertionError if None.

    Returns
    -------
    T
        The provided ``value`` when it is not ``None``.
    """
    return expect_is_not_none(value, label=label, message=message)


def require_row(row: Sequence[object] | None, *, message: str | None = None) -> Sequence[object]:
    """Ensure a fetched row exists and return it for indexing.

    Returns
    -------
    Sequence[object]
        The provided ``row`` when it is not ``None``.
    """
    return expect_is_not_none(row, message=message)


def require_rows[S: Sequence[object]](rows: S | None, *, message: str | None = None) -> S:
    """Ensure a sequence of rows exists and is non-empty.

    Returns
    -------
    S
        The provided ``rows`` when present and non-empty.

    Raises
    ------
    AssertionError
        If ``rows`` is ``None`` or empty.
    """
    seq = expect_is_not_none(rows, message=message)
    if len(seq) == 0:
        raise AssertionError(message or "Expected at least one row")
    return cast("S", seq)


def require_non_empty_sequence[S: Sequence[object]](
    sequence: S | None,
    *,
    message: str | None = None,
) -> S:
    """Ensure a generic sequence exists and has at least one element.

    Returns
    -------
    S
        The provided sequence when present and non-empty.

    Raises
    ------
    AssertionError
        If the sequence is ``None`` or empty.
    """
    seq = expect_is_not_none(sequence, message=message)
    if len(seq) == 0:
        raise AssertionError(message or "Expected non-empty sequence")
    return cast("S", seq)


def expect_row_value(
    row: Sequence[object] | None,
    index: int,
    expected: object,
    *,
    message: str | None = None,
) -> Sequence[object]:
    """Validate a fetched row has an expected value at an index.

    Returns
    -------
    Sequence[object]
        The provided row when it is present and contains the expected value.

    Raises
    ------
    AssertionError
        If the row is missing, too short, or the value does not match.
    """
    seq = require_row(row, message=message)
    if index >= len(seq):
        raise AssertionError(
            message or f"Row length {len(seq)} shorter than expected index {index}"
        )
    expect_equal(seq[index], expected, label=message or f"row[{index}]")
    return seq


def expect_rows_equal(
    rows: Sequence[Sequence[object]] | None,
    expected: Sequence[Sequence[object]],
    *,
    message: str | None = None,
) -> Sequence[Sequence[object]]:
    """Validate a list of rows matches the expected rows.

    Returns
    -------
    Sequence[Sequence[object]]
        The provided rows when present and matching the expected rows.

    Raises
    ------
    AssertionError
        If rows are missing, have different lengths, or differ in content.
    """
    actual = expect_is_not_none(rows, message=message)
    if len(actual) != len(expected):
        raise AssertionError(message or f"Expected {len(expected)} rows, got {len(actual)}")
    for idx, (act_row, exp_row) in enumerate(zip(actual, expected, strict=True)):
        if tuple(act_row) != tuple(exp_row):
            raise AssertionError(message or f"Row {idx} expected {exp_row!r}, got {act_row!r}")
    return actual


__all__ = [
    "expect_empty",
    "expect_equal",
    "expect_false",
    "expect_in",
    "expect_is_instance",
    "expect_is_none",
    "expect_is_not",
    "expect_is_not_none",
    "expect_length",
    "expect_not_empty",
    "expect_not_equal",
    "expect_not_in",
    "expect_row_value",
    "expect_rows_equal",
    "expect_true",
    "require_non_empty_sequence",
    "require_row",
    "require_rows",
    "unwrap_optional",
]
