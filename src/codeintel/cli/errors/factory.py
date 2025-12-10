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

from typing import TYPE_CHECKING

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


def fail_missing_required[T](param: str, *, detail: str | None = None) -> CliResult[T]:
    """Create failed result for missing required parameter.

    Parameters
    ----------
    param
        Parameter name that is missing.
    detail
        Optional detailed message.

    Returns
    -------
    CliResult[T]
        Failed result with validation error.
    """
    return CliResult.fail(
        validation_error(
            ValidationErrorCode.MISSING_REQUIRED,
            param,
            detail or f"Required parameter missing: {param}",
        )
    )


def fail_invalid_value[T](
    param: str,
    value: object,
    reason: str,
    *,
    suggestion: str | None = None,
) -> CliResult[T]:
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
    CliResult[T]
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


def fail_not_found[T](
    resource_type: str,
    identifier: str,
    *,
    detail: str | None = None,
) -> CliResult[T]:
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
    CliResult[T]
        Failed result with not found error.
    """
    return CliResult.fail(
        operation_error(
            OperationErrorCode.NOT_FOUND,
            identifier,
            detail or f"{resource_type.title()} not found: {identifier}",
        )
    )


def fail_storage_connection[T](
    db_path: str | Path,
    cause: str,
) -> CliResult[T]:
    """Create failed result for storage connection failure.

    Parameters
    ----------
    db_path
        Path to the database.
    cause
        Error message from the connection attempt.

    Returns
    -------
    CliResult[T]
        Failed result with storage error.
    """
    return CliResult.fail(
        storage_error(
            StorageErrorCode.CONNECTION_FAILED,
            f"Failed to connect to storage at {db_path}: {cause}",
        )
    )


def fail_storage_query[T](
    message: str,
    *,
    query: str | None = None,
    table: str | None = None,
) -> CliResult[T]:
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
    CliResult[T]
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


def fail_internal[T](
    message: str,
    *,
    operation_id: str = "unknown",
    cause: Exception | None = None,
) -> CliResult[T]:
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
    CliResult[T]
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


def fail_with_problem[T](
    error_type: str,
    title: str,
    detail: str,
    *,
    status: int = 400,
) -> CliResult[T]:
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
    CliResult[T]
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


# -----------------------------------------------------------------------------
# Domain-Specific Error Factories
# -----------------------------------------------------------------------------


def fail_job_not_found[T](job_id: str) -> CliResult[T]:
    """Create failed result for job not found.

    Parameters
    ----------
    job_id
        Job identifier that was not found.

    Returns
    -------
    CliResult[T]
        Failed result with job not found error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:not-found",
            title="Job Not Found",
            detail=f"Job not found: {job_id}",
            status=404,
        )
    )


def fail_job_not_completed[T](job_id: str, current_status: str) -> CliResult[T]:
    """Create failed result for job not in completed state.

    Parameters
    ----------
    job_id
        Job identifier.
    current_status
        Current status of the job.

    Returns
    -------
    CliResult[T]
        Failed result with job not completed error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:not-completed",
            title="Job Not Completed",
            detail=f"Job {job_id} is not completed (status: {current_status})",
            status=400,
        )
    )


def fail_job_cancel_failed[T](job_id: str) -> CliResult[T]:
    """Create failed result for job cancellation failure.

    Parameters
    ----------
    job_id
        Job identifier that could not be cancelled.

    Returns
    -------
    CliResult[T]
        Failed result with cancel failure error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:cancel-failed",
            title="Cancel Failed",
            detail=f"Could not cancel job {job_id}",
            status=400,
        )
    )


def fail_operation_not_found[T](op_id: str) -> CliResult[T]:
    """Create failed result for operation not found.

    Parameters
    ----------
    op_id
        Operation identifier that was not found.

    Returns
    -------
    CliResult[T]
        Failed result with operation not found error.
    """
    return fail_not_found("operation", op_id)


def fail_dataset_not_found[T](table_key: str) -> CliResult[T]:
    """Create failed result for dataset not found.

    Parameters
    ----------
    table_key
        Dataset table key that was not found.

    Returns
    -------
    CliResult[T]
        Failed result with dataset not found error.
    """
    return fail_not_found("dataset", table_key)


def fail_macro_validation[T](message: str, *, missing: list[str] | None = None) -> CliResult[T]:
    """Create failed result for macro validation failure.

    Parameters
    ----------
    message
        Validation failure message.
    missing
        Optional list of missing macro names.

    Returns
    -------
    CliResult[T]
        Failed result with macro validation error.
    """
    detail = message
    if missing:
        detail = f"{message}. Missing macros: {', '.join(missing)}"

    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:storage:macro-validation-failed",
            title="Macro Validation Failed",
            detail=detail,
            status=400,
        )
    )


def fail_invalid_param_format[T](
    param: str, expected: str, got: str | None = None
) -> CliResult[T]:
    """Create failed result for invalid parameter format.

    Parameters
    ----------
    param
        Parameter name.
    expected
        Expected format description.
    got
        Optional actual value received.

    Returns
    -------
    CliResult[T]
        Failed result with invalid format error.
    """
    detail = f"Invalid format for '{param}': expected {expected}"
    if got:
        detail = f"{detail}, got: {got}"

    return CliResult.fail(
        validation_error(
            ValidationErrorCode.INVALID_FORMAT,
            param,
            detail,
        )
    )


__all__ = [
    "fail_dataset_not_found",
    "fail_internal",
    "fail_invalid_param_format",
    "fail_invalid_value",
    "fail_job_cancel_failed",
    "fail_job_not_completed",
    "fail_job_not_found",
    "fail_macro_validation",
    "fail_missing_required",
    "fail_not_found",
    "fail_operation_not_found",
    "fail_storage_connection",
    "fail_storage_query",
    "fail_with_problem",
]
