"""Convenience error factories for CLI handlers.

Provide high-level error construction that returns CliResult directly,
making error handling more concise in handlers.

These factory functions return ``CliResult.fail()`` directly, making them
convenient for use as return values in handlers. They use the standard
error taxonomy under the hood.

Example
-------
>>> from codeintel.cli.errors.factory import fail_missing_required
>>> def my_handler(ctx):
...     if not ctx.param_str("name"):
...         return fail_missing_required("name")
...     # ... rest of handler
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Never

from codeintel.cli.core.results import CliResult
from codeintel.cli.errors._cli_errors import ProblemDetail
from codeintel.cli.errors.taxonomy import (
    OperationErrorCode,
    StorageErrorCode,
    ValidationErrorCode,
    operation_error,
    storage_error,
    validation_error,
)

if TYPE_CHECKING:
    from pathlib import Path


def fail_missing_required(param: str, *, detail: str | None = None) -> CliResult[Never]:
    """Create failed result for missing required parameter.

    Parameters
    ----------
    param
        Parameter name that is missing.
    detail
        Optional detailed message.

    Returns
    -------
    CliResult[Never]
        Failed result with validation error.
    """
    return CliResult.fail(
        validation_error(
            ValidationErrorCode.MISSING_REQUIRED,
            param,
            detail or f"Required parameter missing: {param}",
        )
    )


def fail_invalid_value(
    param: str,
    value: object,
    reason: str,
    *,
    suggestion: str | None = None,
) -> CliResult[Never]:
    """Create failed result for invalid parameter value.

    Parameters
    ----------
    param
        Parameter name.
    value
        The invalid value.
    reason
        Why the value is invalid.
    suggestion
        Optional suggestion for valid values.

    Returns
    -------
    CliResult[Never]
        Failed result with validation error.
    """
    return CliResult.fail(
        validation_error(
            ValidationErrorCode.INVALID_FORMAT,
            param,
            reason,
            value=value,
            suggestion=suggestion,
        )
    )


def fail_not_found(
    resource_type: str,
    identifier: str,
    *,
    detail: str | None = None,
) -> CliResult[Never]:
    """Create failed result for resource not found.

    Parameters
    ----------
    resource_type
        Type of resource (e.g., 'operation', 'dataset', 'target').
    identifier
        Resource identifier.
    detail
        Optional detailed message.

    Returns
    -------
    CliResult[Never]
        Failed result with not found error.
    """
    return CliResult.fail(
        operation_error(
            OperationErrorCode.NOT_FOUND,
            identifier,
            detail or f"{resource_type.title()} not found: {identifier}",
        )
    )


def fail_storage_connection(
    db_path: str | Path,
    cause: str,
) -> CliResult[Never]:
    """Create failed result for storage connection failure.

    Parameters
    ----------
    db_path
        Path to the database.
    cause
        Error message from the connection attempt.

    Returns
    -------
    CliResult[Never]
        Failed result with storage error.
    """
    return CliResult.fail(
        storage_error(
            StorageErrorCode.CONNECTION_FAILED,
            f"Failed to connect to storage at {db_path}: {cause}",
        )
    )


def fail_storage_query(
    message: str,
    *,
    query: str | None = None,
    table: str | None = None,
) -> CliResult[Never]:
    """Create failed result for storage query failure.

    Parameters
    ----------
    message
        Error message.
    query
        The failing query.
    table
        The table involved.

    Returns
    -------
    CliResult[Never]
        Failed result with storage error.
    """
    return CliResult.fail(
        storage_error(
            StorageErrorCode.QUERY_FAILED,
            message,
            query=query,
            table=table,
        )
    )


def fail_internal(
    message: str,
    *,
    operation_id: str = "unknown",
    cause: Exception | None = None,
) -> CliResult[Never]:
    """Create failed result for internal/unexpected failures.

    Parameters
    ----------
    message
        Error message.
    operation_id
        Operation that failed.
    cause
        Underlying exception.

    Returns
    -------
    CliResult[Never]
        Failed result with internal error.
    """
    return CliResult.fail(
        operation_error(
            OperationErrorCode.INTERNAL_ERROR,
            operation_id,
            message,
            cause=cause,
        )
    )


def fail_with_problem(
    error_type: str,
    title: str,
    detail: str,
    *,
    status: int = 400,
) -> CliResult[Never]:
    """Create failed result from explicit ProblemDetail fields.

    Use this for domain-specific errors that don't fit the standard categories.
    Prefer the typed factories above for common error patterns.

    Parameters
    ----------
    error_type
        Error type URI (e.g., 'urn:codeintel:build:invalid-target').
    title
        Short, human-readable summary.
    detail
        Detailed error description.
    status
        HTTP-style status code.

    Returns
    -------
    CliResult[Never]
        Failed result with the specified error.
    """
    return CliResult.fail(
        ProblemDetail(
            type=error_type,
            title=title,
            detail=detail,
            status=status,
        )
    )


__all__ = [
    "fail_internal",
    "fail_invalid_value",
    "fail_missing_required",
    "fail_not_found",
    "fail_storage_connection",
    "fail_storage_query",
    "fail_with_problem",
]
