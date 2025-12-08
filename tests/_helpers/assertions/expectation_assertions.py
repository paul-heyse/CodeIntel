"""Expectation-style assertion helpers for test validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast


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


__all__ = [
    "expect_empty",
    "expect_equal",
    "expect_false",
    "expect_in",
    "expect_is_instance",
    "expect_is_none",
    "expect_is_not_none",
    "expect_length",
    "expect_not_empty",
    "expect_not_equal",
    "expect_not_in",
    "expect_true",
    "require_row",
    "unwrap_optional",
]
