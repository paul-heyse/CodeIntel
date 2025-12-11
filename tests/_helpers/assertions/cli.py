"""CLI handler assertion helpers.

This module provides assertion functions for CLI handler test results.
These helpers follow the same patterns as other assertion modules in
the test suite, providing clear error messages and consistent APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.cli.core import CliResult
    from codeintel.cli.core.results import ProblemDetail


def _prefix(label: str | None) -> str:
    """Build optional label prefix for failure messages.

    Returns
    -------
    str
        Label with colon/space suffix, or empty string if no label.
    """
    return f"{label}: " if label else ""


def expect_handler_success[T](result: CliResult[T], *, message: str | None = None) -> T:
    """Assert that a handler result indicates success.

    Parameters
    ----------
    result
        The CliResult to check.
    message
        Optional custom assertion message.

    Returns
    -------
    T
        The result data (unwrapped).

    Raises
    ------
    AssertionError
        If the result indicates failure or data is None.

    Examples
    --------
    >>> result = my_handler(ctx)  # doctest: +SKIP
    >>> data = expect_handler_success(result)
    """
    msg = message or "Expected handler to succeed"
    if not result.success:
        error_info = ""
        if result.error is not None:
            error_info = f": {result.error.type} - {result.error.title}"
        failure_message = f"{msg}{error_info}"
        raise AssertionError(failure_message)
    if result.data is None:
        failure_message = f"{msg}: success but data is None"
        raise AssertionError(failure_message)
    return result.data


def expect_handler_failure(result: CliResult[Any], *, message: str | None = None) -> ProblemDetail:
    """Assert that a handler result indicates failure.

    Parameters
    ----------
    result
        The CliResult to check.
    message
        Optional custom assertion message.

    Returns
    -------
    ProblemDetail
        The error details from the result.

    Raises
    ------
    AssertionError
        If the result indicates success or error is None.

    Examples
    --------
    >>> result = my_handler(ctx)  # doctest: +SKIP
    >>> error = expect_handler_failure(result)
    """
    msg = message or "Expected handler to fail"
    if result.success:
        raise AssertionError(msg)
    if result.error is None:
        failure_message = f"{msg}: failure but error is None"
        raise AssertionError(failure_message)
    return result.error


def expect_handler_error(
    result: CliResult[Any],
    *,
    error_type: str | None = None,
    status: int | None = None,
    title_contains: str | None = None,
    message: str | None = None,
) -> ProblemDetail:
    """Assert that a handler result has a specific error.

    Parameters
    ----------
    result
        The CliResult to check.
    error_type
        Expected error type URN (exact match).
    status
        Expected HTTP status code.
    title_contains
        Substring that must appear in the error title.
    message
        Optional custom assertion message prefix.

    Returns
    -------
    ProblemDetail
        The error details from the result.

    Raises
    ------
    AssertionError
        If the result doesn't match expectations.

    Examples
    --------
    >>> result = my_handler(ctx)  # doctest: +SKIP
    >>> error = expect_handler_error(result, status=400)
    """
    error = expect_handler_failure(result, message=message)
    msg_prefix = _prefix(message)

    if error_type is not None and error.type != error_type:
        failure_message = f"{msg_prefix}Expected error type '{error_type}', got '{error.type}'"
        raise AssertionError(failure_message)

    if status is not None and error.status != status:
        failure_message = f"{msg_prefix}Expected status {status}, got {error.status}"
        raise AssertionError(failure_message)

    if title_contains is not None and title_contains not in (error.title or ""):
        failure_message = (
            f"{msg_prefix}Expected title containing '{title_contains}', got '{error.title}'"
        )
        raise AssertionError(failure_message)

    return error


def expect_handler_data_count[T](
    result: CliResult[T],
    key: str,
    expected: int,
    *,
    message: str | None = None,
) -> None:
    """Assert that a handler result data has a specific count field.

    Parameters
    ----------
    result
        The CliResult to check.
    key
        The attribute name on the data object containing the count.
    expected
        Expected count value.
    message
        Optional custom assertion message.

    Raises
    ------
    AssertionError
        If the result doesn't match expectations.

    Examples
    --------
    >>> result = list_handler(ctx)  # doctest: +SKIP
    >>> expect_handler_data_count(result, "count", 5)
    """
    data = expect_handler_success(result, message=message)
    actual = getattr(data, key, None)
    msg_prefix = _prefix(message)
    if actual is None:
        failure_message = f"{msg_prefix}Data has no attribute '{key}'"
        raise AssertionError(failure_message)
    if actual != expected:
        failure_message = f"{msg_prefix}Expected {key}={expected}, got {actual}"
        raise AssertionError(failure_message)


def expect_handler_data_contains[T](
    result: CliResult[T],
    key: str,
    expected_item: object,
    *,
    message: str | None = None,
) -> None:
    """Assert that a handler result data list contains an expected item.

    Parameters
    ----------
    result
        The CliResult to check.
    key
        The attribute name on the data object containing the list.
    expected_item
        Item that should be in the list.
    message
        Optional custom assertion message.

    Raises
    ------
    AssertionError
        If the list doesn't contain the expected item.
    """
    data = expect_handler_success(result, message=message)
    actual_list = getattr(data, key, None)
    msg_prefix = _prefix(message)
    if actual_list is None:
        failure_message = f"{msg_prefix}Data has no attribute '{key}'"
        raise AssertionError(failure_message)
    if expected_item not in actual_list:
        failure_message = (
            f"{msg_prefix}Expected {key} to contain {expected_item!r}, got {actual_list!r}"
        )
        raise AssertionError(failure_message)


def expect_handler_warnings(
    result: CliResult[Any],
    *,
    min_count: int | None = None,
    contains: str | None = None,
    message: str | None = None,
) -> list[str]:
    """Assert expectations about handler warnings.

    Parameters
    ----------
    result
        The CliResult to check.
    min_count
        Minimum number of warnings expected.
    contains
        Substring that at least one warning must contain.
    message
        Optional custom assertion message prefix.

    Returns
    -------
    list[str]
        The warnings from the result.

    Raises
    ------
    AssertionError
        If warnings don't match expectations.
    """
    msg_prefix = _prefix(message)
    warnings = result.warnings

    if min_count is not None and len(warnings) < min_count:
        failure_message = f"{msg_prefix}Expected at least {min_count} warnings, got {len(warnings)}"
        raise AssertionError(failure_message)

    if contains is not None:
        matching = [w for w in warnings if contains in w]
        if not matching:
            failure_message = (
                f"{msg_prefix}Expected warning containing '{contains}', got {warnings}"
            )
            raise AssertionError(failure_message)

    return warnings


def expect_handler_metadata(
    result: CliResult[Any],
    key: str,
    expected: object,
    *,
    message: str | None = None,
) -> None:
    """Assert that handler result metadata contains an expected value.

    Parameters
    ----------
    result
        The CliResult to check.
    key
        Metadata key to check.
    expected
        Expected value.
    message
        Optional custom assertion message.

    Raises
    ------
    AssertionError
        If metadata doesn't match.
    """
    msg_prefix = _prefix(message)
    actual = result.metadata.get(key)
    if actual != expected:
        failure_message = f"{msg_prefix}Expected metadata['{key}']={expected!r}, got {actual!r}"
        raise AssertionError(failure_message)


__all__ = [
    "expect_handler_data_contains",
    "expect_handler_data_count",
    "expect_handler_error",
    "expect_handler_failure",
    "expect_handler_metadata",
    "expect_handler_success",
    "expect_handler_warnings",
]
