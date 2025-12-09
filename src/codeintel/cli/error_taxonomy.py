"""Standardized error types and codes for CLI operations.

This module defines the error taxonomy and provides factory functions
for creating RFC 9457 Problem Details with consistent structure.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

from codeintel.cli.cli_errors import ProblemDetail


class ErrorCategory(Enum):
    """Top-level error categories."""

    VALIDATION = "validation-error"
    OPERATION = "operation-error"
    STORAGE = "storage-error"
    CONFIG = "config-error"
    SERVICE = "service-error"


class ValidationErrorCode(Enum):
    """Validation error codes."""

    MISSING_REQUIRED = "missing-required"
    INVALID_TYPE = "invalid-type"
    INVALID_FORMAT = "invalid-format"
    OUT_OF_RANGE = "out-of-range"


class OperationErrorCode(Enum):
    """Operation error codes."""

    NOT_FOUND = "not-found"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILED = "dependency-failed"
    INTERNAL_ERROR = "internal-error"


class StorageErrorCode(Enum):
    """Storage error codes."""

    CONNECTION_FAILED = "connection-failed"
    QUERY_FAILED = "query-failed"
    SCHEMA_MISMATCH = "schema-mismatch"


class ConfigErrorCode(Enum):
    """Configuration error codes."""

    FILE_NOT_FOUND = "file-not-found"
    PARSE_ERROR = "parse-error"
    INVALID_VALUE = "invalid-value"


class ServiceErrorCode(Enum):
    """External service error codes."""

    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate-limited"
    AUTHENTICATION_FAILED = "authentication-failed"


# HTTP status code mappings
_STATUS_CODES: dict[str, int] = {
    ValidationErrorCode.MISSING_REQUIRED.value: 400,
    ValidationErrorCode.INVALID_TYPE.value: 400,
    ValidationErrorCode.INVALID_FORMAT.value: 400,
    ValidationErrorCode.OUT_OF_RANGE.value: 400,
    OperationErrorCode.NOT_FOUND.value: 404,
    OperationErrorCode.TIMEOUT.value: 504,
    OperationErrorCode.DEPENDENCY_FAILED.value: 424,
    OperationErrorCode.INTERNAL_ERROR.value: 500,
    StorageErrorCode.CONNECTION_FAILED.value: 503,
    StorageErrorCode.QUERY_FAILED.value: 500,
    StorageErrorCode.SCHEMA_MISMATCH.value: 500,
    ConfigErrorCode.FILE_NOT_FOUND.value: 404,
    ConfigErrorCode.PARSE_ERROR.value: 400,
    ConfigErrorCode.INVALID_VALUE.value: 400,
    ServiceErrorCode.UNAVAILABLE.value: 503,
    ServiceErrorCode.RATE_LIMITED.value: 429,
    ServiceErrorCode.AUTHENTICATION_FAILED.value: 401,
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


__all__ = [
    "ConfigErrorCode",
    "ErrorCategory",
    "ErrorContext",
    "OperationErrorCode",
    "ServiceErrorCode",
    "StorageErrorCode",
    "ValidationErrorCode",
    "config_error",
    "make_error_type",
    "operation_error",
    "service_error",
    "storage_error",
    "validation_error",
]
