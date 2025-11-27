"""Minimal expectation helpers to avoid assert statements in tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def _prefix(label: str | None) -> str:
    return f"{label}: " if label else ""


def expect_true(condition: object, *, message: str | None = None) -> None:
    """
    Raise AssertionError when condition is falsy.

    Raises
    ------
    AssertionError
        If ``condition`` evaluates to ``False``.
    """
    if bool(condition):
        return
    failure_message = message or "Expected condition to be true"
    raise AssertionError(failure_message)


def expect_equal(actual: T, expected: T, *, label: str | None = None) -> None:
    """
    Assert equality with an optional label.

    Raises
    ------
    AssertionError
        If ``actual`` and ``expected`` differ.
    """
    if actual == expected:
        return
    failure_message = f"{_prefix(label)}expected {expected!r}, got {actual!r}"
    raise AssertionError(failure_message)


def expect_in(member: T, container: Iterable[T], *, label: str | None = None) -> None:
    """
    Assert that member is present in container.

    Raises
    ------
    AssertionError
        If ``member`` is not found.
    """
    if member in container:
        return
    failure_message = f"{_prefix(label)}{member!r} not found in {container!r}"
    raise AssertionError(failure_message)


def expect_is_instance(
    value: object,
    expected_type: type[object],
    *,
    label: str | None = None,
) -> None:
    """
    Assert that a value is an instance of a type.

    Raises
    ------
    AssertionError
        If ``value`` is not an instance of ``expected_type``.
    """
    if isinstance(value, expected_type):
        return
    failure_message = f"{_prefix(label)}expected instance of {expected_type!r}, got {type(value)!r}"
    raise AssertionError(failure_message)


def expect_length(sequence: Iterable[object], expected: int, *, label: str | None = None) -> None:
    """
    Assert that a sequence has the expected length.

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
