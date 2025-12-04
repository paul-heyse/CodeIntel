"""Expectation-style assertion helpers for test validation."""

from __future__ import annotations

from collections.abc import Iterable


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
    if value not in container:
        failure_message = f"{_prefix(label)}{value!r} not found in {container!r}"
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


__all__ = [
    "expect_equal",
    "expect_in",
    "expect_is_instance",
    "expect_length",
    "expect_true",
]
