"""Standardized error types and codes for CLI operations.

This module emits CLI-scoped RFC 9457 Problem `type` URNs using the
`urn:codeintel:cli:` namespace.

The CLI taxonomy follows this hierarchy:

    urn:codeintel:cli:
    ├── validation:
    │   ├── missing-required
    │   ├── invalid-type
    │   ├── invalid-format
    │   ├── out-of-range
    │   └── constraint-violation
    ├── operation:
    │   ├── not-found
    │   ├── already-exists
    │   ├── timeout
    │   ├── dependency-failed
    │   ├── cancelled
    │   └── internal-error
    ├── storage:
    │   ├── connection-failed
    │   ├── query-failed
    │   ├── schema-mismatch
    │   └── corruption-detected
    ├── config:
    │   ├── file-not-found
    │   ├── parse-error
    │   ├── invalid-value
    │   └── schema-violation
    ├── service:
    │   ├── unavailable
    │   ├── rate-limited
    │   ├── authentication-failed
    │   └── permission-denied
    └── job:
        ├── not-found
        ├── already-running
        ├── failed
        └── expired
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

# Use CLI-specific ProblemDetail for rendering
from codeintel.cli.errors._cli_errors import ProblemDetail
from codeintel.core.errors.taxonomy import (
    CONFIG_FILE_NOT_FOUND,
    CONFIG_SCHEMA_VIOLATION,
    INTERNAL_ERROR,
    JOB_NOT_FOUND,
    MISSING_REQUIRED,
    NOT_FOUND,
    TIMEOUT,
    ErrorCategory,
    ErrorCode,
    OperationErrorCode,
)


def make_error_type(category: ErrorCategory, code: str) -> str:
    """Construct a CLI-scoped error type URN.

    Parameters
    ----------
    category
        Error category.
    code
        Category-specific error code value.

    Returns
    -------
    str
        RFC 9457 `type` identifier for CLI errors.
    """
    return f"urn:codeintel:cli:{category.value}:{code}"


def validation_error(
    code: ErrorCode,
    field_name: str,
    message: str,
    *,
    value: object = None,
    suggestion: str | None = None,
) -> ProblemDetail:
    """Create a validation error.

    Parameters
    ----------
    code
        Validation error code descriptor.
    field_name
        Field that failed validation.
    message
        Error message.
    value
        The invalid value (if safe to include).
    suggestion
        Suggestion for fixing the error.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {"field": field_name}
    if value is not None:
        extensions["value"] = str(value)[:100]
    if suggestion:
        extensions["suggestion"] = suggestion

    return ProblemDetail(
        type=make_error_type(code.category, code.code),
        title="Validation Error",
        detail=message,
        status=code.status,
        extensions=extensions,
    )


def operation_error(
    code: ErrorCode,
    operation_id: str,
    message: str,
    *,
    cause: Exception | None = None,
    debug_info: dict[str, Any] | None = None,
) -> ProblemDetail:
    """Create an operation error.

    Parameters
    ----------
    code
        Operation error code descriptor.
    operation_id
        The operation that failed.
    message
        Error message.
    cause
        Underlying exception (included in debug mode).
    debug_info
        Additional debug information.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {"operation_id": operation_id}
    if cause is not None:
        extensions["cause_type"] = type(cause).__name__
    if debug_info:
        extensions["debug"] = debug_info

    return ProblemDetail(
        type=make_error_type(code.category, code.code),
        title="Operation Error",
        detail=message,
        status=code.status,
        instance=f"/operations/{operation_id}",
        extensions=extensions,
    )


def storage_error(
    code: ErrorCode,
    message: str,
    *,
    query: str | None = None,
    table: str | None = None,
) -> ProblemDetail:
    """Create a storage error.

    Parameters
    ----------
    code
        Storage error code descriptor.
    message
        Error message.
    query
        The failing query (truncated).
    table
        The table involved.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {}
    if query:
        extensions["query"] = query[:200]
    if table:
        extensions["table"] = table

    return ProblemDetail(
        type=make_error_type(code.category, code.code),
        title="Storage Error",
        detail=message,
        status=code.status,
        extensions=extensions if extensions else {},
    )


def config_error(
    code: ErrorCode,
    message: str,
    *,
    path: str | None = None,
    key: str | None = None,
) -> ProblemDetail:
    """Create a configuration error.

    Parameters
    ----------
    code
        Configuration error code descriptor.
    message
        Error message.
    path
        Config file path.
    key
        Config key that failed.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {}
    if path:
        extensions["path"] = path
    if key:
        extensions["key"] = key

    return ProblemDetail(
        type=make_error_type(code.category, code.code),
        title="Configuration Error",
        detail=message,
        status=code.status,
        extensions=extensions if extensions else {},
    )


def service_error(
    code: ErrorCode,
    service: str,
    message: str,
    *,
    retry_after: float | None = None,
) -> ProblemDetail:
    """Create a service error.

    Parameters
    ----------
    code
        Service error code descriptor.
    service
        Name of the failing service.
    message
        Error message.
    retry_after
        Seconds to wait before retry.

    Returns
    -------
    ProblemDetail
        Structured error.
    """
    extensions: dict[str, Any] = {"service": service}
    if retry_after is not None:
        extensions["retry_after_seconds"] = retry_after

    return ProblemDetail(
        type=make_error_type(code.category, code.code),
        title="Service Error",
        detail=message,
        status=code.status,
        extensions=extensions,
    )


@dataclass
class ErrorContext:
    """Context for error creation with debug support.

    Parameters
    ----------
    debug_mode
        Whether to include debug information.
    correlation_id
        Request correlation ID.
    """

    debug_mode: bool = False
    correlation_id: str | None = None

    def wrap_exception(
        self,
        exc: Exception,
        *,
        operation_id: str | None = None,
    ) -> ProblemDetail:
        """Wrap an exception as a ProblemDetail.

        Parameters
        ----------
        exc
            Exception to wrap.
        operation_id
            Optional operation context.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        debug_info: dict[str, Any] | None = None
        if self.debug_mode:
            debug_info = {
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }

        extensions: dict[str, Any] = {}
        if self.correlation_id:
            extensions["correlation_id"] = self.correlation_id
        if debug_info:
            extensions["debug"] = debug_info

        error_type = make_error_type(
            ErrorCategory.OPERATION,
            OperationErrorCode.INTERNAL_ERROR.value,
        )

        detail = str(exc) if self.debug_mode else "An unexpected error occurred"

        return ProblemDetail(
            type=error_type,
            title="Internal Error",
            detail=detail,
            status=500,
            instance=f"/operations/{operation_id}" if operation_id else None,
            extensions=extensions if extensions else {},
        )


@dataclass
class StructuredCliError(Exception):
    """Base CLI error with Problem Detail support.

    This exception class provides structured error information that can
    be converted to RFC 9457 Problem Details for consistent error handling.

    Parameters
    ----------
    error_code
        Error code with metadata.
    detail
        Human-readable detail message.
    extensions
        Additional context data.
    suggestion
        Suggested fix (optional).
    cause
        Underlying exception (optional).
    """

    error_code: ErrorCode
    detail: str
    extensions: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None
    cause: Exception | None = None

    def __post_init__(self) -> None:
        """Initialize the exception with the detail message."""
        super().__init__(self.detail)

    def to_problem_detail(self, *, debug: bool = False) -> ProblemDetail:
        """Convert to RFC 9457 Problem Detail.

        Parameters
        ----------
        debug
            Include debug information like stack traces.

        Returns
        -------
        ProblemDetail
            Structured error for rendering.
        """
        ext = dict(self.extensions)
        if self.suggestion:
            ext["suggestion"] = self.suggestion
        if debug and self.cause:
            ext["cause_type"] = type(self.cause).__name__
            ext["cause_message"] = str(self.cause)
            ext["traceback"] = traceback.format_exception(self.cause)

        return ProblemDetail(
            type=self.error_code.type_uri,
            title=self.error_code.title,
            detail=self.detail,
            status=self.error_code.status,
            extensions=ext if ext else {},
        )


class StructuredValidationError(StructuredCliError):
    """Validation-specific error with field context.

    Parameters
    ----------
    error_code
        Validation error code.
    field_name
        Field that failed validation.
    message
        Error message.
    value
        The invalid value (optional).
    suggestion
        Suggested fix (optional).
    """

    field_name: str = ""
    value: object = None

    def __init__(
        self,
        error_code: ErrorCode,
        field_name: str,
        message: str,
        *,
        value: object = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize validation error."""
        self.field_name = field_name
        self.value = value
        extensions: dict[str, Any] = {"field": field_name}
        if value is not None:
            extensions["value"] = str(value)[:100]
        super().__init__(
            error_code=error_code,
            detail=f"{field_name}: {message}",
            extensions=extensions,
            suggestion=suggestion,
        )


class StructuredOperationError(StructuredCliError):
    """Operation-specific error with operation context.

    Parameters
    ----------
    error_code
        Operation error code.
    operation_id
        Operation that failed.
    message
        Error message.
    cause
        Underlying exception (optional).
    suggestion
        Suggested fix (optional).
    """

    operation_id: str = ""

    def __init__(
        self,
        error_code: ErrorCode,
        operation_id: str,
        message: str,
        *,
        cause: Exception | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize operation error."""
        self.operation_id = operation_id
        extensions: dict[str, Any] = {"operation_id": operation_id}
        super().__init__(
            error_code=error_code,
            detail=message,
            extensions=extensions,
            suggestion=suggestion,
            cause=cause,
        )


def missing_required_error(
    field_name: str,
    *,
    suggestion: str | None = None,
) -> StructuredValidationError:
    """Create a missing required parameter error.

    Parameters
    ----------
    field_name
        Name of the missing field.
    suggestion
        Suggested fix.

    Returns
    -------
    StructuredValidationError
        Validation error for missing required parameter.
    """
    return StructuredValidationError(
        MISSING_REQUIRED,
        field_name,
        "This parameter is required",
        suggestion=suggestion or f"Provide a value for --{field_name}",
    )


def not_found_error(
    resource_type: str,
    resource_id: str,
    *,
    suggestion: str | None = None,
) -> StructuredCliError:
    """Create a resource not found error.

    Parameters
    ----------
    resource_type
        Type of resource (e.g., "operation", "job").
    resource_id
        Identifier of the missing resource.
    suggestion
        Suggested fix.

    Returns
    -------
    StructuredCliError
        Error for resource not found.
    """
    return StructuredCliError(
        error_code=NOT_FOUND,
        detail=f"{resource_type} not found: {resource_id}",
        extensions={"resource_type": resource_type, "resource_id": resource_id},
        suggestion=suggestion,
    )


def timeout_error(
    operation_id: str,
    timeout_seconds: float,
    *,
    suggestion: str | None = None,
) -> StructuredOperationError:
    """Create an operation timeout error.

    Parameters
    ----------
    operation_id
        Operation that timed out.
    timeout_seconds
        Timeout duration.
    suggestion
        Suggested fix.

    Returns
    -------
    StructuredOperationError
        Error for operation timeout.
    """
    return StructuredOperationError(
        TIMEOUT,
        operation_id,
        f"Operation timed out after {timeout_seconds}s",
        suggestion=suggestion or "Try increasing timeout or breaking into smaller operations",
    )


def internal_error(
    message: str,
    *,
    operation_id: str | None = None,
    cause: Exception | None = None,
) -> StructuredCliError:
    """Create an internal error.

    Parameters
    ----------
    message
        Error message.
    operation_id
        Operation context (optional).
    cause
        Underlying exception.

    Returns
    -------
    StructuredCliError
        Internal error.
    """
    extensions: dict[str, Any] = {}
    if operation_id:
        extensions["operation_id"] = operation_id

    return StructuredCliError(
        error_code=INTERNAL_ERROR,
        detail=message,
        extensions=extensions,
        cause=cause,
    )


def job_not_found_error(job_id: str) -> StructuredCliError:
    """Create a job not found error.

    Parameters
    ----------
    job_id
        Job identifier.

    Returns
    -------
    StructuredCliError
        Job not found error.
    """
    return StructuredCliError(
        error_code=JOB_NOT_FOUND,
        detail=f"Job not found: {job_id}",
        extensions={"job_id": job_id},
        suggestion="Use 'codeintel jobs list' to see available jobs",
    )


def config_not_found_error(path: str) -> StructuredCliError:
    """Create a configuration file not found error.

    Parameters
    ----------
    path
        Path to the missing config file.

    Returns
    -------
    StructuredCliError
        Config file not found error.
    """
    return StructuredCliError(
        error_code=CONFIG_FILE_NOT_FOUND,
        detail=f"Configuration file not found: {path}",
        extensions={"path": path},
        suggestion="Use 'codeintel config init' to create a config file",
    )


def config_validation_error(
    errors: list[str],
    *,
    path: str | None = None,
) -> StructuredCliError:
    """Create a configuration validation error.

    Parameters
    ----------
    errors
        List of validation error messages.
    path
        Config file path.

    Returns
    -------
    StructuredCliError
        Config validation error.
    """
    extensions: dict[str, Any] = {"errors": errors}
    if path:
        extensions["path"] = path

    return StructuredCliError(
        error_code=CONFIG_SCHEMA_VIOLATION,
        detail=f"Configuration validation failed: {len(errors)} error(s)",
        extensions=extensions,
    )


__all__ = [
    "ErrorContext",
    "StructuredCliError",
    "StructuredOperationError",
    "StructuredValidationError",
    "config_error",
    "config_not_found_error",
    "config_validation_error",
    "internal_error",
    "job_not_found_error",
    "make_error_type",
    "missing_required_error",
    "not_found_error",
    "operation_error",
    "service_error",
    "storage_error",
    "timeout_error",
    "validation_error",
]
