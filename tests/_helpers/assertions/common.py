"""Shared assertion helpers for plugin results.

This module provides protocol-based assertion helpers that work with both
analytics and ingestion plugin results. The assertions are designed to be
type-safe and provide clear error messages.

Example
-------
>>> from tests._helpers.assertions import (
...     assert_row_count,
...     assert_success,
...     format_assertion_message,
... )
>>> assert_success(result.success, result.error)
>>> assert_row_count(result.row_counts, "analytics.my_table", min_rows=1)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping


@runtime_checkable
class HasRowCounts(Protocol):
    """Protocol for results with row counts."""

    @property
    def row_counts(self) -> Mapping[str, int] | None:
        """Return row counts by table name."""
        ...


@runtime_checkable
class HasSuccessAndError(Protocol):
    """Protocol for results with success/error fields."""

    @property
    def success(self) -> bool:
        """Return whether execution succeeded."""
        ...

    @property
    def error(self) -> str | None:
        """Return error message if any."""
        ...


def format_assertion_message(prefix: str, message: str) -> str:
    """Format an assertion message with optional prefix.

    Parameters
    ----------
    prefix
        Optional prefix to prepend.
    message
        The main message content.

    Returns
    -------
    str
        Formatted message.
    """
    return f"{prefix}{message}".strip()


def assert_success(
    *,
    success: bool,
    error: str | None,
    message_prefix: str = "",
) -> None:
    """Assert that execution succeeded.

    Parameters
    ----------
    success
        Whether execution succeeded.
    error
        Error message if any.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If execution failed.
    """
    if not success:
        msg = format_assertion_message(
            message_prefix,
            f"Expected success but got failure: {error}",
        )
        raise AssertionError(msg)


def assert_failure(
    *,
    success: bool,
    message_prefix: str = "",
) -> None:
    """Assert that execution failed.

    Parameters
    ----------
    success
        Whether execution succeeded.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If execution succeeded.
    """
    if success:
        msg = format_assertion_message(
            message_prefix,
            "Expected failure but got success",
        )
        raise AssertionError(msg)


def assert_has_error(
    error: str | None,
    *,
    containing: str | None = None,
    message_prefix: str = "",
) -> None:
    """Assert that there is an error message.

    Parameters
    ----------
    error
        Error message to check.
    containing
        Optional substring the error must contain.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If no error or substring not found.
    """
    if error is None:
        msg = format_assertion_message(message_prefix, "Expected error but got none")
        raise AssertionError(msg)

    if containing is not None and containing not in error:
        msg = format_assertion_message(
            message_prefix,
            f"Expected error containing '{containing}' but got: {error}",
        )
        raise AssertionError(msg)


def assert_no_error(
    error: str | None,
    *,
    message_prefix: str = "",
) -> None:
    """Assert that there is no error message.

    Parameters
    ----------
    error
        Error message to check.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If there is an error.
    """
    if error is not None:
        msg = format_assertion_message(
            message_prefix,
            f"Expected no error but got: {error}",
        )
        raise AssertionError(msg)


def assert_row_count(
    row_counts: Mapping[str, int] | None,
    table: str,
    *,
    min_rows: int | None = None,
    max_rows: int | None = None,
    exact: int | None = None,
) -> None:
    """Assert row count for a table.

    Parameters
    ----------
    row_counts
        Mapping of table names to row counts.
    table
        Table name to check.
    min_rows
        Minimum expected rows.
    max_rows
        Maximum expected rows.
    exact
        Exact expected row count.

    Raises
    ------
    AssertionError
        If row count doesn't match expectations.
    """
    counts = row_counts or {}
    actual = counts.get(table, 0)

    if exact is not None:
        if actual != exact:
            msg = f"Expected {table} to have {exact} rows, got {actual}"
            raise AssertionError(msg)
        return

    if min_rows is not None and actual < min_rows:
        msg = f"Expected {table} to have at least {min_rows} rows, got {actual}"
        raise AssertionError(msg)

    if max_rows is not None and actual > max_rows:
        msg = f"Expected {table} to have at most {max_rows} rows, got {actual}"
        raise AssertionError(msg)


def assert_meta_contains(
    meta: Mapping[str, object],
    key: str,
    *,
    value: object | None = None,
    message_prefix: str = "",
) -> None:
    """Assert that metadata contains a key.

    Parameters
    ----------
    meta
        Metadata mapping to check.
    key
        Key that must be present.
    value
        Optional expected value.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If key missing or value doesn't match.
    """
    if key not in meta:
        msg = format_assertion_message(
            message_prefix,
            f"Expected meta to contain '{key}'",
        )
        raise AssertionError(msg)

    if value is not None and meta[key] != value:
        msg = format_assertion_message(
            message_prefix,
            f"Expected meta['{key}'] to be {value!r}, got {meta[key]!r}",
        )
        raise AssertionError(msg)


def assert_valid(
    errors: list[str],
    *,
    valid: bool,
    message_prefix: str = "",
) -> None:
    """Assert that validation passed.

    Parameters
    ----------
    valid
        Whether validation passed.
    errors
        List of validation errors.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If validation failed.
    """
    if not valid:
        msg = format_assertion_message(
            message_prefix,
            f"Expected valid but got errors: {errors}",
        )
        raise AssertionError(msg)


def assert_invalid(
    *,
    valid: bool,
    message_prefix: str = "",
) -> None:
    """Assert that validation failed.

    Parameters
    ----------
    valid
        Whether validation passed.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If validation passed.
    """
    if valid:
        msg = format_assertion_message(
            message_prefix,
            "Expected invalid but validation passed",
        )
        raise AssertionError(msg)


def assert_validation_error(
    errors: list[str],
    containing: str,
    *,
    message_prefix: str = "",
) -> None:
    """Assert that there is a validation error containing text.

    Parameters
    ----------
    errors
        List of validation errors.
    containing
        Substring to find in errors.
    message_prefix
        Optional prefix for assertion messages.

    Raises
    ------
    AssertionError
        If no error contains the text.
    """
    for error in errors:
        if containing in error:
            return

    msg = format_assertion_message(
        message_prefix,
        f"Expected error containing '{containing}' but got: {errors}",
    )
    raise AssertionError(msg)


def assert_no_subprocess_usage(
    src_root: Path,
    *,
    allowed: Iterable[Path] | None = None,
    patterns: tuple[str, ...] = ("create_subprocess_exec(", "subprocess.run(", "Popen("),
) -> None:
    """Assert no subprocess usage outside the allowed set.

    Raises
    ------
    AssertionError
        If subprocess-related calls are found outside the allowed set.
    """
    repo_root = Path().resolve()
    allowed_set = set(allowed or ())
    violations: list[str] = []
    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(repo_root)
        if rel_path in allowed_set:
            continue
        content = path.read_text(encoding="utf8")
        if any(pattern in content for pattern in patterns):
            violations.append(str(rel_path))
    if violations:
        message = f"Direct subprocess usage found in: {', '.join(sorted(violations))}"
        raise AssertionError(message)


SUBPROCESS_ALLOWLIST: tuple[Path, ...] = (
    Path("src/codeintel/ingestion/engine/infrastructure/runner.py"),
    Path("src/codeintel/ingestion/engine/service.py"),
    Path("src/codeintel/build/providers.py"),
    Path("src/codeintel/cli/jobs/_jobs.py"),
)


__all__ = [
    "SUBPROCESS_ALLOWLIST",
    "HasRowCounts",
    "HasSuccessAndError",
    "assert_failure",
    "assert_has_error",
    "assert_invalid",
    "assert_meta_contains",
    "assert_no_error",
    "assert_no_subprocess_usage",
    "assert_row_count",
    "assert_success",
    "assert_valid",
    "assert_validation_error",
    "format_assertion_message",
]
