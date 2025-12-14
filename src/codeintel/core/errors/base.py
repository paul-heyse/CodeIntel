"""Base error classes for CodeIntel.

This module provides the canonical base exception classes that support
RFC 9457 Problem Details conversion.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from codeintel.core.errors.problem_details import ProblemDetail
from codeintel.core.errors.taxonomy import (
    INTERNAL_ERROR,
    ErrorCategory,
    ErrorCode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class CodeIntelError(Exception):
    """Base exception with Problem Details support.

    All domain-specific exceptions in CodeIntel should inherit from this
    class to ensure consistent error handling and structured output.

    Attributes
    ----------
    error_code
        Structured error code with category and metadata.
    detail
        Human-readable detail message.
    extensions
        Additional context data for debugging.
    suggestion
        Optional suggested fix for the error.
    cause
        Original exception that caused this error.

    Examples
    --------
    >>> from codeintel.core.errors.taxonomy import INTERNAL_ERROR
    >>> error = CodeIntelError(
    ...     error_code=INTERNAL_ERROR,
    ...     detail="Database connection failed",
    ...     extensions={"database": "main"},
    ... )
    >>> problem = error.to_problem_detail()
    >>> problem.status
    500
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
            status=self.error_code.status,
            detail=self.detail,
            extensions=ext if ext else {},
        )

    @property
    def type_uri(self) -> str:
        """Return the error type URI.

        Returns
        -------
        str
            URN for this error type.
        """
        return self.error_code.type_uri

    @property
    def status(self) -> int:
        """Return the HTTP/exit status code.

        Returns
        -------
        int
            Status code.
        """
        return self.error_code.status


class CodeIntelValidationError(CodeIntelError):
    """Validation-specific error with field context.

    Use this for input validation failures where a specific field
    or parameter is invalid.

    Attributes
    ----------
    field_name
        Name of the field that failed validation.
    value
        The invalid value (optional).
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
        """Initialize validation error.

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


class CodeIntelOperationError(CodeIntelError):
    """Operation-specific error with operation context.

    Use this for errors during named operations like commands,
    plugins, or background jobs.

    Attributes
    ----------
    operation_id
        Identifier of the failed operation.
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
        """Initialize operation error.

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
        self.operation_id = operation_id
        extensions: dict[str, Any] = {"operation_id": operation_id}
        super().__init__(
            error_code=error_code,
            detail=message,
            extensions=extensions,
            suggestion=suggestion,
            cause=cause,
        )


class CodeIntelStorageError(CodeIntelError):
    """Storage-specific error with query context.

    Use this for database query failures, connection errors,
    and schema issues.

    Attributes
    ----------
    table
        Table involved in the error.
    query
        SQL query that failed (truncated).
    """

    table: str | None = None
    query: str | None = None

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        *,
        table: str | None = None,
        query: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize storage error.

        Parameters
        ----------
        error_code
            Storage error code.
        message
            Error message.
        table
            Table involved.
        query
            SQL query that failed (truncated to 200 chars).
        cause
            Underlying exception.
        """
        self.table = table
        self.query = query[:200] if query else None
        extensions: dict[str, Any] = {}
        if table:
            extensions["table"] = table
        if query:
            extensions["query"] = query[:200]
        super().__init__(
            error_code=error_code,
            detail=message,
            extensions=extensions,
            cause=cause,
        )


@dataclass
class ErrorContext:
    """Context for error creation with debug support.

    Use this to wrap exceptions with consistent problem detail
    generation across a request/operation boundary.

    Attributes
    ----------
    debug_mode
        Whether to include debug information.
    correlation_id
        Request correlation ID for tracing.
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

        error_type = f"urn:codeintel:{ErrorCategory.OPERATION.value}/internal-error"
        detail = str(exc) if self.debug_mode else "An unexpected error occurred"

        return ProblemDetail(
            type=error_type,
            title=INTERNAL_ERROR.title,
            status=INTERNAL_ERROR.status,
            detail=detail,
            instance=f"/operations/{operation_id}" if operation_id else None,
            extensions=extensions if extensions else {},
        )


def aggregate_errors(errors: Sequence[CodeIntelError]) -> ProblemDetail:
    """Aggregate multiple errors into a single problem detail.

    Parameters
    ----------
    errors
        Sequence of errors to aggregate.

    Returns
    -------
    ProblemDetail
        Aggregated problem detail with nested error details.
    """
    if not errors:
        return ProblemDetail(
            type="urn:codeintel:validation/no-errors",
            title="No Errors",
            status=200,
        )

    if len(errors) == 1:
        return errors[0].to_problem_detail()

    # Use the most severe status code
    max_status = max(e.error_code.status for e in errors)
    error_details = [{"type": e.error_code.type_uri, "detail": e.detail} for e in errors]

    return ProblemDetail(
        type="urn:codeintel:validation/multiple-errors",
        title="Multiple Errors",
        status=max_status,
        detail=f"{len(errors)} errors occurred",
        extensions={"errors": error_details},
    )


__all__ = [
    "CodeIntelError",
    "CodeIntelOperationError",
    "CodeIntelStorageError",
    "CodeIntelValidationError",
    "ErrorContext",
    "aggregate_errors",
]
