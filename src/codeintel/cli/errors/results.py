"""Result factory functions for common error patterns.

Provide high-level result factories that return CliResult.fail() directly,
making error handling concise in handlers.

These factories use ProblemBuilder internally and provide the most common
error patterns. For domain-specific or uncommon errors, use ProblemBuilder
directly or the domain() method.

Examples
--------
>>> from codeintel.cli.errors.results import fail_validation, fail_not_found
>>> def my_handler(ctx):
...     if not ctx.param_str("name"):
...         return fail_validation("name", "Parameter is required")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.cli.core.results import CliResult
from codeintel.cli.errors.builder import ProblemBuilder
from codeintel.core.errors.taxonomy import (
    ConfigErrorCode,
    JobErrorCode,
    OperationErrorCode,
    StorageErrorCode,
    ValidationErrorCode,
)

if TYPE_CHECKING:
    from pathlib import Path

type ErrorResult = CliResult[Any]


def fail_validation(
    field: str,
    detail: str,
    *,
    code: ValidationErrorCode = ValidationErrorCode.MISSING_REQUIRED,
    value: object = None,
    suggestion: str | None = None,
) -> ErrorResult:
    """Create failed result for validation error.

    Parameters
    ----------
    field
        Field that failed validation.
    detail
        Error message.
    code
        Specific validation error code.
    value
        The invalid value.
    suggestion
        Suggestion for fixing.

    Returns
    -------
    CliResult
        Failed result with validation error.
    """
    return CliResult.fail(
        ProblemBuilder.validation(code, field, detail, value=value, suggestion=suggestion)
    )


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
    return fail_validation(
        param,
        detail or f"Required parameter missing: {param}",
        code=ValidationErrorCode.MISSING_REQUIRED,
        suggestion=f"Provide a value for --{param}",
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
    return fail_validation(
        param,
        reason,
        code=ValidationErrorCode.INVALID_FORMAT,
        value=value,
        suggestion=suggestion,
    )


def fail_not_found(
    resource_type: str,
    identifier: str,
    *,
    suggestion: str | None = None,
) -> ErrorResult:
    """Create failed result for resource not found.

    Parameters
    ----------
    resource_type
        Type of resource (e.g., "operation", "dataset").
    identifier
        Resource identifier.
    suggestion
        Suggestion for finding the resource.

    Returns
    -------
    CliResult
        Failed result with not found error.
    """
    return CliResult.fail(
        ProblemBuilder.not_found(resource_type, identifier, suggestion=suggestion)
    )


def fail_storage(
    detail: str,
    *,
    code: StorageErrorCode = StorageErrorCode.QUERY_FAILED,
    query: str | None = None,
    table: str | None = None,
) -> ErrorResult:
    """Create failed result for storage error.

    Parameters
    ----------
    detail
        Error message.
    code
        Storage error code.
    query
        The failing query.
    table
        The table involved.

    Returns
    -------
    CliResult
        Failed result with storage error.
    """
    return CliResult.fail(ProblemBuilder.storage(code, detail, query=query, table=table))


def fail_storage_connection(db_path: str | Path, cause: str) -> ErrorResult:
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
    return fail_storage(
        f"Failed to connect to storage at {db_path}: {cause}",
        code=StorageErrorCode.CONNECTION_FAILED,
    )


def fail_internal(
    message: str,
    *,
    cause: Exception | None = None,
    operation_id: str | None = None,
) -> ErrorResult:
    """Create failed result for internal/unexpected failures.

    Parameters
    ----------
    message
        Error message.
    cause
        Underlying exception.
    operation_id
        Operation context.

    Returns
    -------
    CliResult
        Failed result with internal error.
    """
    return CliResult.fail(ProblemBuilder.internal(message, cause=cause, operation_id=operation_id))


def fail_operation(
    operation_id: str,
    detail: str,
    *,
    code: OperationErrorCode = OperationErrorCode.INTERNAL_ERROR,
    cause: Exception | None = None,
) -> ErrorResult:
    """Create failed result for operation error.

    Parameters
    ----------
    operation_id
        The operation that failed.
    detail
        Error message.
    code
        Operation error code.
    cause
        Underlying exception.

    Returns
    -------
    CliResult
        Failed result with operation error.
    """
    return CliResult.fail(ProblemBuilder.operation(code, operation_id, detail, cause=cause))


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
        ProblemBuilder.job(JobErrorCode.NOT_FOUND, job_id, f"Job not found: {job_id}")
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
        Failed result with job error.
    """
    return CliResult.fail(
        ProblemBuilder.job(
            JobErrorCode.FAILED,
            job_id,
            f"Job {job_id} is not completed (status: {current_status})",
        )
    )


def fail_domain(
    domain: str,
    code: str,
    title: str,
    detail: str,
    *,
    status: int = 400,
    **extensions: object,
) -> ErrorResult:
    """Create failed result for domain-specific error.

    Parameters
    ----------
    domain
        Domain identifier (e.g., "build", "plugins").
    code
        Error code within the domain.
    title
        Human-readable title.
    detail
        Error detail message.
    status
        HTTP-style status code.
    **extensions
        Additional context fields.

    Returns
    -------
    CliResult
        Failed result with domain error.
    """
    return CliResult.fail(
        ProblemBuilder.domain(domain, code, title, detail, status=status, **extensions)
    )


def fail_config(
    detail: str,
    *,
    code: ConfigErrorCode = ConfigErrorCode.INVALID_VALUE,
    path: str | None = None,
    key: str | None = None,
) -> ErrorResult:
    """Create failed result for configuration error.

    Parameters
    ----------
    detail
        Error message.
    code
        Config error code.
    path
        Config file path.
    key
        Config key that failed.

    Returns
    -------
    CliResult
        Failed result with config error.
    """
    return CliResult.fail(ProblemBuilder.config(code, detail, path=path, key=key))


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
    return fail_storage(message, code=StorageErrorCode.QUERY_FAILED, query=query, table=table)


def fail_with_problem(
    error_type: str,
    title: str,
    detail: str,
    *,
    status: int = 400,
) -> ErrorResult:
    """Create failed result from explicit ProblemDetail fields.

    Use this for domain-specific errors that don't fit the standard categories.
    Prefer the typed factories for common error patterns.

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
    parts = error_type.split(":")

    domain_index = 2
    domain = parts[domain_index] if len(parts) > domain_index else "cli"
    code = parts[-1] if parts else error_type
    return CliResult.fail(ProblemBuilder.domain(domain, code, title, detail, status=status))


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
        ProblemBuilder.job(JobErrorCode.FAILED, job_id, f"Could not cancel job {job_id}")
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

    return fail_domain(
        "storage",
        "macro-validation-failed",
        "Macro Validation Failed",
        detail,
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

    return fail_validation(
        param,
        detail,
        code=ValidationErrorCode.INVALID_FORMAT,
    )


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
    return fail_domain(domain, "project-error", "Project Error", message)


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
    return fail_domain(
        domain,
        "file-not-found",
        "File Not Found",
        f"File not found: {file_path}",
        status=404,
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
    return fail_domain(domain, "execution-failed", "Execution Failed", message, status=status)


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
    return fail_domain(
        "build",
        "invalid-module",
        "Invalid Module",
        f"Unknown module: {module}. Valid: {', '.join(valid_modules)}",
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
    return fail_domain("build", "invalid-selection", "Invalid Target Selection", message)


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
    return fail_domain("build", "invalid-targets", "Invalid Targets", message)


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
    return fail_domain("build", "run-not-found", "Run Not Found", message, status=404)


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
    return fail_domain(
        "plugins", "not-found", "Plugin Not Found", f"Plugin not found: {name}", status=404
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
    return fail_domain("plugins", "invalid-name", "Invalid Plugin Name", reason)


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
    return fail_domain(
        "plugins",
        "no-manifest",
        "No Plugin Manifest",
        f"No plugin.json found in {path}",
        status=404,
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
    return fail_domain(
        "plugins",
        "invalid-manifest",
        "Invalid Plugin Manifest",
        f"Error loading manifest: {message}",
    )


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
    return fail_domain(
        "graphs",
        "invalid-policy",
        f"Invalid {policy_type.title()} Policy",
        f"Unknown {policy_type} policy: {value}",
    )


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
    return fail_domain(
        "ops", "unknown-operation", "Unknown Operation", f"Unknown operation: {op_id}", status=404
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
    return fail_domain(
        "ops",
        "invalid-param",
        "Invalid Parameter Format",
        f"Invalid parameter format: {param_str} (expected key=value)",
    )


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
    return fail_domain("storage", "no-tables", "No Tables Specified", message)


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
    return fail_domain(
        "storage",
        f"missing-{param.replace('_', '-')}",
        f"Missing {param.replace('_', ' ').title()}",
        f"{param} parameter is required.",
    )


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
    return fail_domain(
        "subsystem",
        "not-found",
        "Subsystem not found",
        f"Subsystem not found: {subsystem_id}",
        status=404,
        instance=f"subsystem://{subsystem_id}",
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
    return fail_domain(
        "ide",
        "hints-not-found",
        "No hints found",
        f"No IDE hints found for: {rel_path}",
        status=404,
        instance=f"file://{rel_path}",
    )


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
    return fail_domain(
        "history",
        title.lower().replace(" ", "-"),
        title,
        detail,
        status=status,
    )


__all__ = [
    "fail_build_run_not_found",
    "fail_config",
    "fail_dataset_not_found",
    "fail_domain",
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
    "fail_operation",
    "fail_operation_not_found",
    "fail_plugin_no_manifest",
    "fail_plugin_not_found",
    "fail_project_error",
    "fail_storage",
    "fail_storage_connection",
    "fail_storage_query",
    "fail_subsystem_not_found",
    "fail_unknown_operation",
    "fail_validation",
    "fail_with_problem",
]
