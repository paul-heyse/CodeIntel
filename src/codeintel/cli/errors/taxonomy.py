"""Standardized error types and codes for CLI operations.

This module defines the error taxonomy and provides factory functions
for creating RFC 9457 Problem Details with consistent structure.

The error taxonomy follows this hierarchy:

    urn:codeintel:cli:
    ├── validation/
    │   ├── missing-required
    │   ├── invalid-type
    │   ├── invalid-format
    │   ├── out-of-range
    │   └── constraint-violation
    ├── operation/
    │   ├── not-found
    │   ├── already-exists
    │   ├── timeout
    │   ├── dependency-failed
    │   ├── cancelled
    │   └── internal-error
    ├── storage/
    │   ├── connection-failed
    │   ├── query-failed
    │   ├── schema-mismatch
    │   └── corruption-detected
    ├── config/
    │   ├── file-not-found
    │   ├── parse-error
    │   ├── invalid-value
    │   └── schema-violation
    ├── service/
    │   ├── unavailable
    │   ├── rate-limited
    │   ├── authentication-failed
    │   └── permission-denied
    └── job/
        ├── not-found
        ├── already-running
        ├── failed
        └── expired
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from codeintel.cli.errors._cli_errors import ProblemDetail


class ErrorCategory(Enum):
    """Top-level error categories."""

    VALIDATION = "validation"
    OPERATION = "operation"
    STORAGE = "storage"
    CONFIG = "config"
    SERVICE = "service"
    JOB = "job"


class ValidationErrorCode(Enum):
    """Validation error codes."""

    MISSING_REQUIRED = "missing-required"
    INVALID_TYPE = "invalid-type"
    INVALID_FORMAT = "invalid-format"
    OUT_OF_RANGE = "out-of-range"
    CONSTRAINT_VIOLATION = "constraint-violation"


class OperationErrorCode(Enum):
    """Operation error codes."""

    NOT_FOUND = "not-found"
    ALREADY_EXISTS = "already-exists"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILED = "dependency-failed"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal-error"


class StorageErrorCode(Enum):
    """Storage error codes."""

    CONNECTION_FAILED = "connection-failed"
    QUERY_FAILED = "query-failed"
    SCHEMA_MISMATCH = "schema-mismatch"
    CORRUPTION_DETECTED = "corruption-detected"


class ConfigErrorCode(Enum):
    """Configuration error codes."""

    FILE_NOT_FOUND = "file-not-found"
    PARSE_ERROR = "parse-error"
    INVALID_VALUE = "invalid-value"
    SCHEMA_VIOLATION = "schema-violation"


class ServiceErrorCode(Enum):
    """External service error codes."""

    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate-limited"
    AUTHENTICATION_FAILED = "authentication-failed"
    PERMISSION_DENIED = "permission-denied"


class JobErrorCode(Enum):
    """Background job error codes."""

    NOT_FOUND = "not-found"
    ALREADY_RUNNING = "already-running"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ErrorCode:
    """Structured error code with metadata.

    Parameters
    ----------
    category
        Error category.
    code
        Specific error code.
    status
        HTTP status code.
    title
        Human-readable title.
    """

    category: ErrorCategory
    code: str
    status: int
    title: str

    @property
    def type_uri(self) -> str:
        """Get fully-qualified error type URI.

        Returns
        -------
        str
            Error type URI in format urn:codeintel:cli:category/code.
        """
        return f"urn:codeintel:cli:{self.category.value}/{self.code}"


# Validation Error Codes
MISSING_REQUIRED = ErrorCode(
    ErrorCategory.VALIDATION, "missing-required", 400, "Missing Required Parameter"
)
INVALID_TYPE = ErrorCode(ErrorCategory.VALIDATION, "invalid-type", 400, "Invalid Parameter Type")
INVALID_FORMAT = ErrorCode(
    ErrorCategory.VALIDATION, "invalid-format", 400, "Invalid Parameter Format"
)
OUT_OF_RANGE = ErrorCode(ErrorCategory.VALIDATION, "out-of-range", 400, "Value Out of Range")
CONSTRAINT_VIOLATION = ErrorCode(
    ErrorCategory.VALIDATION, "constraint-violation", 400, "Constraint Violation"
)

# Operation Error Codes
NOT_FOUND = ErrorCode(ErrorCategory.OPERATION, "not-found", 404, "Resource Not Found")
ALREADY_EXISTS = ErrorCode(
    ErrorCategory.OPERATION, "already-exists", 409, "Resource Already Exists"
)
TIMEOUT = ErrorCode(ErrorCategory.OPERATION, "timeout", 504, "Operation Timeout")
DEPENDENCY_FAILED = ErrorCode(
    ErrorCategory.OPERATION, "dependency-failed", 424, "Dependency Failed"
)
CANCELLED = ErrorCode(ErrorCategory.OPERATION, "cancelled", 499, "Operation Cancelled")
INTERNAL_ERROR = ErrorCode(ErrorCategory.OPERATION, "internal-error", 500, "Internal Error")

# Storage Error Codes
CONNECTION_FAILED = ErrorCode(
    ErrorCategory.STORAGE, "connection-failed", 503, "Storage Connection Failed"
)
QUERY_FAILED = ErrorCode(ErrorCategory.STORAGE, "query-failed", 500, "Query Failed")
SCHEMA_MISMATCH = ErrorCode(ErrorCategory.STORAGE, "schema-mismatch", 500, "Schema Mismatch")
CORRUPTION_DETECTED = ErrorCode(
    ErrorCategory.STORAGE, "corruption-detected", 500, "Data Corruption Detected"
)

# Config Error Codes
CONFIG_FILE_NOT_FOUND = ErrorCode(
    ErrorCategory.CONFIG, "file-not-found", 404, "Configuration File Not Found"
)
CONFIG_PARSE_ERROR = ErrorCode(
    ErrorCategory.CONFIG, "parse-error", 400, "Configuration Parse Error"
)
CONFIG_INVALID_VALUE = ErrorCode(
    ErrorCategory.CONFIG, "invalid-value", 400, "Invalid Configuration Value"
)
CONFIG_SCHEMA_VIOLATION = ErrorCode(
    ErrorCategory.CONFIG, "schema-violation", 400, "Configuration Schema Violation"
)

# Service Error Codes
SERVICE_UNAVAILABLE = ErrorCode(ErrorCategory.SERVICE, "unavailable", 503, "Service Unavailable")
RATE_LIMITED = ErrorCode(ErrorCategory.SERVICE, "rate-limited", 429, "Rate Limited")
AUTH_FAILED = ErrorCode(
    ErrorCategory.SERVICE, "authentication-failed", 401, "Authentication Failed"
)
PERMISSION_DENIED = ErrorCode(ErrorCategory.SERVICE, "permission-denied", 403, "Permission Denied")

# Job Error Codes
JOB_NOT_FOUND = ErrorCode(ErrorCategory.JOB, "not-found", 404, "Job Not Found")
JOB_ALREADY_RUNNING = ErrorCode(ErrorCategory.JOB, "already-running", 409, "Job Already Running")
JOB_FAILED = ErrorCode(ErrorCategory.JOB, "failed", 500, "Job Execution Failed")
JOB_EXPIRED = ErrorCode(ErrorCategory.JOB, "expired", 410, "Job Results Expired")

# HTTP status code mappings for error code enums
_STATUS_CODES: dict[str, int] = {
    ValidationErrorCode.MISSING_REQUIRED.value: 400,
    ValidationErrorCode.INVALID_TYPE.value: 400,
    ValidationErrorCode.INVALID_FORMAT.value: 400,
    ValidationErrorCode.OUT_OF_RANGE.value: 400,
    ValidationErrorCode.CONSTRAINT_VIOLATION.value: 400,
    OperationErrorCode.NOT_FOUND.value: 404,
    OperationErrorCode.ALREADY_EXISTS.value: 409,
    OperationErrorCode.TIMEOUT.value: 504,
    OperationErrorCode.DEPENDENCY_FAILED.value: 424,
    OperationErrorCode.CANCELLED.value: 499,
    OperationErrorCode.INTERNAL_ERROR.value: 500,
    StorageErrorCode.CONNECTION_FAILED.value: 503,
    StorageErrorCode.QUERY_FAILED.value: 500,
    StorageErrorCode.SCHEMA_MISMATCH.value: 500,
    StorageErrorCode.CORRUPTION_DETECTED.value: 500,
    ConfigErrorCode.FILE_NOT_FOUND.value: 404,
    ConfigErrorCode.PARSE_ERROR.value: 400,
    ConfigErrorCode.INVALID_VALUE.value: 400,
    ConfigErrorCode.SCHEMA_VIOLATION.value: 400,
    ServiceErrorCode.UNAVAILABLE.value: 503,
    ServiceErrorCode.RATE_LIMITED.value: 429,
    ServiceErrorCode.AUTHENTICATION_FAILED.value: 401,
    ServiceErrorCode.PERMISSION_DENIED.value: 403,
    JobErrorCode.NOT_FOUND.value: 404,
    JobErrorCode.ALREADY_RUNNING.value: 409,
    JobErrorCode.FAILED.value: 500,
    JobErrorCode.EXPIRED.value: 410,
}


def make_error_type(category: ErrorCategory, code: str) -> str:
    """Create a fully-qualified error type URI.

    Parameters
    ----------
    category
        Error category.
    code
        Specific error code.

    Returns
    -------
    str
        Error type URI.
    """
    return f"urn:codeintel:cli:{category.value}:{code}"


def validation_error(
    code: ValidationErrorCode,
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
        Validation error code.
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
        extensions["value"] = str(value)[:100]  # Truncate for safety
    if suggestion:
        extensions["suggestion"] = suggestion

    return ProblemDetail(
        type=make_error_type(ErrorCategory.VALIDATION, code.value),
        title="Validation Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 400),
        extensions=extensions,
    )


def operation_error(
    code: OperationErrorCode,
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
        Operation error code.
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
        type=make_error_type(ErrorCategory.OPERATION, code.value),
        title="Operation Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 500),
        instance=f"/operations/{operation_id}",
        extensions=extensions,
    )


def storage_error(
    code: StorageErrorCode,
    message: str,
    *,
    query: str | None = None,
    table: str | None = None,
) -> ProblemDetail:
    """Create a storage error.

    Parameters
    ----------
    code
        Storage error code.
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
        extensions["query"] = query[:200]  # Truncate
    if table:
        extensions["table"] = table

    return ProblemDetail(
        type=make_error_type(ErrorCategory.STORAGE, code.value),
        title="Storage Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 500),
        extensions=extensions if extensions else {},
    )


def config_error(
    code: ConfigErrorCode,
    message: str,
    *,
    path: str | None = None,
    key: str | None = None,
) -> ProblemDetail:
    """Create a configuration error.

    Parameters
    ----------
    code
        Config error code.
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
        type=make_error_type(ErrorCategory.CONFIG, code.value),
        title="Configuration Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 400),
        extensions=extensions if extensions else {},
    )


def service_error(
    code: ServiceErrorCode,
    service: str,
    message: str,
    *,
    retry_after: float | None = None,
) -> ProblemDetail:
    """Create a service error.

    Parameters
    ----------
    code
        Service error code.
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
        type=make_error_type(ErrorCategory.SERVICE, code.value),
        title="Service Error",
        detail=message,
        status=_STATUS_CODES.get(code.value, 503),
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


# Factory functions for common errors


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
    "ALREADY_EXISTS",
    "AUTH_FAILED",
    "CANCELLED",
    "CONFIG_FILE_NOT_FOUND",
    "CONFIG_INVALID_VALUE",
    "CONFIG_PARSE_ERROR",
    "CONFIG_SCHEMA_VIOLATION",
    "CONNECTION_FAILED",
    "CONSTRAINT_VIOLATION",
    "CORRUPTION_DETECTED",
    "DEPENDENCY_FAILED",
    "INTERNAL_ERROR",
    "INVALID_FORMAT",
    "INVALID_TYPE",
    "JOB_ALREADY_RUNNING",
    "JOB_EXPIRED",
    "JOB_FAILED",
    "JOB_NOT_FOUND",
    "MISSING_REQUIRED",
    "NOT_FOUND",
    "OUT_OF_RANGE",
    "PERMISSION_DENIED",
    "QUERY_FAILED",
    "RATE_LIMITED",
    "SCHEMA_MISMATCH",
    "SERVICE_UNAVAILABLE",
    "TIMEOUT",
    "ConfigErrorCode",
    "ErrorCategory",
    "ErrorCode",
    "ErrorContext",
    "JobErrorCode",
    "OperationErrorCode",
    "ServiceErrorCode",
    "StorageErrorCode",
    "StructuredCliError",
    "StructuredOperationError",
    "StructuredValidationError",
    "ValidationErrorCode",
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
