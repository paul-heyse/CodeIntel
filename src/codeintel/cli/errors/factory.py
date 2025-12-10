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

from typing import TYPE_CHECKING, Any

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

type ErrorResult = CliResult[Any]


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
    CliResult
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
    CliResult
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
    CliResult
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
    CliResult
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
    CliResult
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
    CliResult
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
    CliResult
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


def fail_job_not_found(job_id: str) -> ErrorResult:
    """Create failed result for job not found.

    Parameters
    ----------
    job_id
        Job identifier that was not found.

    Returns
    -------
    CliResult
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
    CliResult
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


def fail_job_cancel_failed(job_id: str) -> ErrorResult:
    """Create failed result for job cancellation failure.

    Parameters
    ----------
    job_id
        Job identifier that could not be cancelled.

    Returns
    -------
    CliResult
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


def fail_operation_not_found(op_id: str) -> ErrorResult:
    """Create failed result for operation not found.

    Parameters
    ----------
    op_id
        Operation identifier that was not found.

    Returns
    -------
    CliResult
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
    CliResult
        Failed result with dataset not found error.
    """
    return fail_not_found("dataset", table_key)


def fail_macro_validation(message: str, *, missing: list[str] | None = None) -> ErrorResult:
    """Create failed result for macro validation failure.

    Parameters
    ----------
    message
        Validation failure message.
    missing
        Optional list of missing macro names.

    Returns
    -------
    CliResult
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


def fail_invalid_param_format(param: str, expected: str, got: str | None = None) -> ErrorResult:
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
    CliResult
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


# -----------------------------------------------------------------------------
# Project/Resolution Error Factories
# -----------------------------------------------------------------------------


def fail_project_error(domain: str, message: str) -> ErrorResult:
    """Create failed result for project resolution errors.

    Parameters
    ----------
    domain
        Domain identifier (e.g., 'build', 'datasets', 'docs').
    message
        Error message from the resolution failure.

    Returns
    -------
    CliResult
        Failed result with project error.
    """
    return CliResult.fail(
        ProblemDetail(
            type=f"urn:codeintel:{domain}:project-error",
            title="Project Error",
            detail=message,
            status=400,
        )
    )


def fail_file_not_found(file_path: str, *, domain: str = "storage") -> ErrorResult:
    """Create failed result for file not found.

    Parameters
    ----------
    file_path
        Path to the file that was not found.
    domain
        Domain identifier for the error type.

    Returns
    -------
    CliResult
        Failed result with file not found error.
    """
    return CliResult.fail(
        ProblemDetail(
            type=f"urn:codeintel:{domain}:file-not-found",
            title="File Not Found",
            detail=f"File not found: {file_path}",
            status=404,
        )
    )


def fail_execution_failed(domain: str, message: str, *, status: int = 500) -> ErrorResult:
    """Create failed result for execution failures.

    Parameters
    ----------
    domain
        Domain identifier (e.g., 'build', 'analytics').
    message
        Error message from the execution failure.
    status
        HTTP-style status code (default 500).

    Returns
    -------
    CliResult
        Failed result with execution error.
    """
    return CliResult.fail(
        ProblemDetail(
            type=f"urn:codeintel:{domain}:execution-failed",
            title="Execution Failed",
            detail=message,
            status=status,
        )
    )


# -----------------------------------------------------------------------------
# Build Domain Error Factories
# -----------------------------------------------------------------------------


def fail_invalid_module(module: str, valid_modules: tuple[str, ...]) -> ErrorResult:
    """Create failed result for invalid module name.

    Parameters
    ----------
    module
        The invalid module name provided.
    valid_modules
        Tuple of valid module names.

    Returns
    -------
    CliResult
        Failed result with invalid module error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:build:invalid-module",
            title="Invalid Module",
            detail=f"Unknown module: {module}. Valid: {', '.join(valid_modules)}",
            status=400,
        )
    )


def fail_invalid_target_selection(message: str) -> ErrorResult:
    """Create failed result for invalid target selection.

    Parameters
    ----------
    message
        Description of the selection error.

    Returns
    -------
    CliResult
        Failed result with invalid selection error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:build:invalid-selection",
            title="Invalid Target Selection",
            detail=message,
            status=400,
        )
    )


def fail_invalid_targets(message: str) -> ErrorResult:
    """Create failed result for invalid build targets.

    Parameters
    ----------
    message
        Error message about the invalid targets.

    Returns
    -------
    CliResult
        Failed result with invalid targets error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:build:invalid-targets",
            title="Invalid Targets",
            detail=message,
            status=400,
        )
    )


def fail_build_run_not_found(message: str) -> ErrorResult:
    """Create failed result for build run not found.

    Parameters
    ----------
    message
        Error message about the missing run.

    Returns
    -------
    CliResult
        Failed result with run not found error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:build:run-not-found",
            title="Run Not Found",
            detail=message,
            status=404,
        )
    )


# -----------------------------------------------------------------------------
# Plugin Domain Error Factories
# -----------------------------------------------------------------------------


def fail_plugin_not_found(name: str) -> ErrorResult:
    """Create failed result for plugin not found.

    Parameters
    ----------
    name
        Plugin name that was not found.

    Returns
    -------
    CliResult
        Failed result with plugin not found error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:plugins:not-found",
            title="Plugin Not Found",
            detail=f"Plugin not found: {name}",
            status=404,
        )
    )


def fail_invalid_plugin_name(reason: str) -> ErrorResult:
    """Create failed result for invalid plugin name.

    Parameters
    ----------
    reason
        Description of why the name is invalid.

    Returns
    -------
    CliResult
        Failed result with invalid plugin name error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:plugins:invalid-name",
            title="Invalid Plugin Name",
            detail=reason,
            status=400,
        )
    )


def fail_plugin_no_manifest(path: str) -> ErrorResult:
    """Create failed result for missing plugin manifest.

    Parameters
    ----------
    path
        Plugin directory path where manifest was expected.

    Returns
    -------
    CliResult
        Failed result with no manifest error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:plugins:no-manifest",
            title="No Plugin Manifest",
            detail=f"No plugin.json found in {path}",
            status=404,
        )
    )


def fail_invalid_plugin_manifest(message: str) -> ErrorResult:
    """Create failed result for invalid plugin manifest.

    Parameters
    ----------
    message
        Error message from manifest parsing.

    Returns
    -------
    CliResult
        Failed result with invalid manifest error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:plugins:invalid-manifest",
            title="Invalid Plugin Manifest",
            detail=f"Error loading manifest: {message}",
            status=400,
        )
    )


# -----------------------------------------------------------------------------
# Graph Domain Error Factories
# -----------------------------------------------------------------------------


def fail_invalid_policy(policy_type: str, value: str) -> ErrorResult:
    """Create failed result for invalid policy value.

    Parameters
    ----------
    policy_type
        Type of policy (e.g., 'selection', 'dependency').
    value
        The invalid policy value provided.

    Returns
    -------
    CliResult
        Failed result with invalid policy error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:graphs:invalid-policy",
            title=f"Invalid {policy_type.title()} Policy",
            detail=f"Unknown {policy_type} policy: {value}",
            status=400,
        )
    )


# -----------------------------------------------------------------------------
# Operation/Serving Error Factories
# -----------------------------------------------------------------------------


def fail_unknown_operation(op_id: str) -> ErrorResult:
    """Create failed result for unknown serving operation.

    Parameters
    ----------
    op_id
        Operation ID that was not found.

    Returns
    -------
    CliResult
        Failed result with unknown operation error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:ops:unknown-operation",
            title="Unknown Operation",
            detail=f"Unknown operation: {op_id}",
            status=404,
        )
    )


def fail_invalid_param(param_str: str) -> ErrorResult:
    """Create failed result for invalid parameter format in CLI.

    Parameters
    ----------
    param_str
        The parameter string that was invalid.

    Returns
    -------
    CliResult
        Failed result with invalid param error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:ops:invalid-param",
            title="Invalid Parameter Format",
            detail=f"Invalid parameter format: {param_str} (expected key=value)",
            status=400,
        )
    )


# -----------------------------------------------------------------------------
# Storage Domain Error Factories
# -----------------------------------------------------------------------------


def fail_no_tables(message: str) -> ErrorResult:
    """Create failed result for no tables specified.

    Parameters
    ----------
    message
        Description of the error.

    Returns
    -------
    CliResult
        Failed result with no tables error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="urn:codeintel:storage:no-tables",
            title="No Tables Specified",
            detail=message,
            status=400,
        )
    )


def fail_missing_output_path(param: str) -> ErrorResult:
    """Create failed result for missing output path parameter.

    Parameters
    ----------
    param
        Name of the missing output parameter.

    Returns
    -------
    CliResult
        Failed result with missing output error.
    """
    return CliResult.fail(
        ProblemDetail(
            type=f"urn:codeintel:storage:missing-{param.replace('_', '-')}",
            title=f"Missing {param.replace('_', ' ').title()}",
            detail=f"{param} parameter is required.",
            status=400,
        )
    )


# -----------------------------------------------------------------------------
# Subsystem/IDE Domain Error Factories
# -----------------------------------------------------------------------------


def fail_subsystem_not_found(subsystem_id: str) -> ErrorResult:
    """Create failed result for subsystem not found.

    Parameters
    ----------
    subsystem_id
        Subsystem ID that was not found.

    Returns
    -------
    CliResult
        Failed result with subsystem not found error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="codeintel:subsystem/not-found",
            title="Subsystem not found",
            status=404,
            detail=f"Subsystem not found: {subsystem_id}",
            instance=f"subsystem://{subsystem_id}",
        )
    )


def fail_ide_hints_not_found(rel_path: str) -> ErrorResult:
    """Create failed result for IDE hints not found.

    Parameters
    ----------
    rel_path
        Relative path that was queried.

    Returns
    -------
    CliResult
        Failed result with hints not found error.
    """
    return CliResult.fail(
        ProblemDetail(
            type="codeintel:ide/hints-not-found",
            title="No hints found",
            status=404,
            detail=f"No IDE hints found for: {rel_path}",
            instance=f"file://{rel_path}",
        )
    )


# -----------------------------------------------------------------------------
# History Domain Error Factories
# -----------------------------------------------------------------------------


def fail_history_error(title: str, detail: str, *, status: int = 1) -> ErrorResult:
    """Create failed result for history command errors.

    Parameters
    ----------
    title
        Short summary of the problem.
    detail
        Detailed error description.
    status
        Exit code for the error (default 1).

    Returns
    -------
    CliResult
        Failed result with history error.
    """
    return CliResult.fail(
        ProblemDetail(
            type=f"urn:codeintel:cli:history:{title.lower().replace(' ', '-')}",
            title=title,
            status=status,
            detail=detail,
        )
    )


__all__ = [
    "fail_build_run_not_found",
    "fail_dataset_not_found",
    "fail_execution_failed",
    "fail_file_not_found",
    "fail_history_error",
    "fail_ide_hints_not_found",
    "fail_internal",
    "fail_invalid_module",
    "fail_invalid_param",
    "fail_invalid_param_format",
    "fail_invalid_plugin_manifest",
    "fail_invalid_plugin_name",
    "fail_invalid_policy",
    "fail_invalid_target_selection",
    "fail_invalid_targets",
    "fail_invalid_value",
    "fail_job_cancel_failed",
    "fail_job_not_completed",
    "fail_job_not_found",
    "fail_macro_validation",
    "fail_missing_output_path",
    "fail_missing_required",
    "fail_no_tables",
    "fail_not_found",
    "fail_operation_not_found",
    "fail_plugin_no_manifest",
    "fail_plugin_not_found",
    "fail_project_error",
    "fail_storage_connection",
    "fail_storage_query",
    "fail_subsystem_not_found",
    "fail_unknown_operation",
    "fail_with_problem",
]
