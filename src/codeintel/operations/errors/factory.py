"""Error factories for operations.

Provide high-level error construction that returns Result directly,
making error handling concise in operation implementations.

These factory functions return ``Result.fail()`` directly, making them
convenient for use as return values in operations.

Example
-------
>>> from codeintel.operations.errors.factory import fail_missing_required
>>> def execute(self, params, ctx):
...     if not params.name:
...         return fail_missing_required("name")
...     # ... rest of implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.operations.errors.problem_detail import ProblemDetail
from codeintel.operations.result import Result

if TYPE_CHECKING:
    from pathlib import Path


ErrorResult = Result[Any]


def fail_missing_required(param: str, *, detail: str | None = None) -> ErrorResult:
    """Create failed result for missing required parameter.

    Parameters
    ----------
    param
        Parameter name that is missing.
    detail
        Optional detailed message.

    Returns
    -------
    Result
        Failed result with validation error.
    """
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:validation:missing-required",
            title="Missing Required Parameter",
            status=400,
            detail=detail or f"Required parameter missing: {param}",
            extensions={"param": param},
        )
    )


def fail_invalid_value(
    param: str,
    value: object,
    reason: str,
    *,
    suggestion: str | None = None,
) -> ErrorResult:
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
    Result
        Failed result with validation error.
    """
    extensions: dict[str, object] = {"param": param, "value": str(value)}
    if suggestion:
        extensions["suggestion"] = suggestion

    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:validation:invalid-value",
            title="Invalid Value",
            status=400,
            detail=reason,
            extensions=extensions,
        )
    )


def fail_not_found(
    resource_type: str,
    identifier: str,
    *,
    detail: str | None = None,
) -> ErrorResult:
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
    Result
        Failed result with not found error.
    """
    return Result.fail(
        ProblemDetail(
            type=f"urn:codeintel:{resource_type.lower()}:not-found",
            title=f"{resource_type.title()} Not Found",
            status=404,
            detail=detail or f"{resource_type.title()} not found: {identifier}",
            instance=f"{resource_type}://{identifier}",
        )
    )


def fail_storage_connection(
    db_path: str | Path,
    cause: str,
) -> ErrorResult:
    """Create failed result for storage connection failure.

    Parameters
    ----------
    db_path
        Path to the database.
    cause
        Error message from the connection attempt.

    Returns
    -------
    Result
        Failed result with storage error.
    """
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:storage:connection-failed",
            title="Storage Connection Failed",
            status=503,
            detail=f"Failed to connect to storage at {db_path}: {cause}",
        )
    )


def fail_storage_query(
    message: str,
    *,
    query: str | None = None,
    table: str | None = None,
) -> ErrorResult:
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
    Result
        Failed result with storage error.
    """
    extensions: dict[str, object] = {}
    if query:
        extensions["query"] = query
    if table:
        extensions["table"] = table

    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:storage:query-failed",
            title="Query Failed",
            status=500,
            detail=message,
            extensions=extensions if extensions else {},
        )
    )


def fail_internal(
    message: str,
    *,
    operation_id: str = "unknown",
    cause: Exception | None = None,
) -> ErrorResult:
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
    Result
        Failed result with internal error.
    """
    extensions: dict[str, object] = {"operation_id": operation_id}
    if cause:
        extensions["exception_type"] = type(cause).__name__

    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:internal:error",
            title="Internal Error",
            status=500,
            detail=message,
            extensions=extensions,
        )
    )


def fail_with_problem(
    error_type: str,
    title: str,
    detail: str,
    *,
    status: int = 400,
) -> ErrorResult:
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
    Result
        Failed result with the specified error.
    """
    return Result.fail(
        ProblemDetail(
            type=error_type,
            title=title,
            detail=detail,
            status=status,
        )
    )


def fail_capability_denied(
    capability: str,
    operation_id: str,
) -> ErrorResult:
    """Create failed result for missing capability.

    Parameters
    ----------
    capability
        Capability that was required.
    operation_id
        Operation that required it.

    Returns
    -------
    Result
        Failed result with capability error.
    """
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:capability:denied",
            title="Capability Denied",
            status=403,
            detail=f"Operation '{operation_id}' requires capability '{capability}'",
            extensions={"capability": capability, "operation_id": operation_id},
        )
    )


# -----------------------------------------------------------------------------
# Domain-Specific Error Factories
# -----------------------------------------------------------------------------


def fail_job_not_found(job_id: str) -> ErrorResult:
    """Create failed result for job not found.

    Parameters
    ----------
    job_id
        Job identifier that was not found.

    Returns
    -------
    Result
        Failed result with job not found error.
    """
    return fail_not_found("job", job_id)


def fail_job_not_completed(job_id: str, current_status: str) -> ErrorResult:
    """Create failed result for job not in completed state.

    Parameters
    ----------
    job_id
        Job identifier.
    current_status
        Current status of the job.

    Returns
    -------
    Result
        Failed result with job not completed error.
    """
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:not-completed",
            title="Job Not Completed",
            detail=f"Job {job_id} is not completed (status: {current_status})",
            status=400,
        )
    )


def fail_job_cancel_failed(job_id: str) -> ErrorResult:
    """Create failed result for job cancellation failure.

    Parameters
    ----------
    job_id
        Job identifier that could not be cancelled.

    Returns
    -------
    Result
        Failed result with cancel failure error.
    """
    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:jobs:cancel-failed",
            title="Cancel Failed",
            detail=f"Could not cancel job {job_id}",
            status=400,
        )
    )


def fail_operation_not_found(op_id: str) -> ErrorResult:
    """Create failed result for operation not found.

    Parameters
    ----------
    op_id
        Operation identifier that was not found.

    Returns
    -------
    Result
        Failed result with operation not found error.
    """
    return fail_not_found("operation", op_id)


def fail_dataset_not_found(table_key: str) -> ErrorResult:
    """Create failed result for dataset not found.

    Parameters
    ----------
    table_key
        Dataset table key that was not found.

    Returns
    -------
    Result
        Failed result with dataset not found error.
    """
    return fail_not_found("dataset", table_key)


def fail_plugin_not_found(name: str) -> ErrorResult:
    """Create failed result for plugin not found.

    Parameters
    ----------
    name
        Plugin name that was not found.

    Returns
    -------
    Result
        Failed result with plugin not found error.
    """
    return fail_not_found("plugin", name)


def fail_validation(message: str, *, param: str | None = None) -> ErrorResult:
    """Create failed result for validation errors.

    Parameters
    ----------
    message
        Validation error message.
    param
        Optional parameter name that failed validation.

    Returns
    -------
    Result
        Failed result with validation error.
    """
    extensions: dict[str, object] = {}
    if param:
        extensions["param"] = param

    return Result.fail(
        ProblemDetail(
            type="urn:codeintel:validation:failed",
            title="Validation Failed",
            status=400,
            detail=message,
            extensions=extensions if extensions else {},
        )
    )


__all__ = [
    "ErrorResult",
    "fail_capability_denied",
    "fail_dataset_not_found",
    "fail_internal",
    "fail_invalid_value",
    "fail_job_cancel_failed",
    "fail_job_not_completed",
    "fail_job_not_found",
    "fail_missing_required",
    "fail_not_found",
    "fail_operation_not_found",
    "fail_plugin_not_found",
    "fail_storage_connection",
    "fail_storage_query",
    "fail_validation",
    "fail_with_problem",
]
