"""Unified error builder for RFC 9457 Problem Details.

Consolidate the 35+ factory functions in factory.py into a streamlined builder
that creates ProblemDetail objects with consistent structure.

The ProblemBuilder provides:
- Category-based error creation (validation, operation, storage, etc.)
- Consistent type URI generation
- Extension handling with truncation for safety
- Debug mode support for exception details
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any

from codeintel.cli.errors.taxonomy import make_error_type
from codeintel.core.errors.problem_details import ProblemDetail
from codeintel.core.errors.taxonomy import (
    INTERNAL_ERROR,
    NOT_FOUND,
    ConfigErrorCode,
    ErrorCategory,
    JobErrorCode,
    OperationErrorCode,
    ServiceErrorCode,
    StorageErrorCode,
    ValidationErrorCode,
)

if TYPE_CHECKING:
    from codeintel.core.errors.taxonomy import ErrorCode


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


class ProblemBuilder:
    """Unified builder for RFC 9457 Problem Details.

    Create ProblemDetail objects with consistent structure across all error
    categories. This consolidates the factory pattern from multiple modules
    into a single, type-safe API.

    Examples
    --------
    >>> error = ProblemBuilder.validation(
    ...     ValidationErrorCode.MISSING_REQUIRED,
    ...     "name",
    ...     "Parameter is required",
    ... )
    >>> error.status
    400
    """

    @classmethod
    def validation(
        cls,
        code: ValidationErrorCode,
        field: str,
        detail: str,
        *,
        value: object = None,
        suggestion: str | None = None,
    ) -> ProblemDetail:
        """Create a validation error.

        Parameters
        ----------
        code
            Validation error code.
        field
            Field that failed validation.
        detail
            Error detail message.
        value
            The invalid value (truncated for safety).
        suggestion
            Suggestion for fixing the error.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        extensions: dict[str, Any] = {"field": field}
        if value is not None:
            extensions["value"] = cls._truncate(str(value), 100)
        if suggestion:
            extensions["suggestion"] = suggestion

        return ProblemDetail(
            type=make_error_type(ErrorCategory.VALIDATION, code.value),
            title="Validation Error",
            detail=detail,
            status=_STATUS_CODES.get(code.value, 400),
            extensions=extensions,
        )

    @classmethod
    def operation(
        cls,
        code: OperationErrorCode,
        operation_id: str,
        detail: str,
        *,
        cause: Exception | None = None,
        debug: bool = False,
    ) -> ProblemDetail:
        """Create an operation error.

        Parameters
        ----------
        code
            Operation error code.
        operation_id
            The operation that failed.
        detail
            Error detail message.
        cause
            Underlying exception.
        debug
            Include debug information.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        extensions: dict[str, Any] = {"operation_id": operation_id}
        if cause is not None:
            extensions["cause_type"] = type(cause).__name__
            if debug:
                extensions["cause_message"] = str(cause)
                extensions["traceback"] = traceback.format_exc()

        return ProblemDetail(
            type=make_error_type(ErrorCategory.OPERATION, code.value),
            title="Operation Error",
            detail=detail,
            status=_STATUS_CODES.get(code.value, 500),
            instance=f"/operations/{operation_id}",
            extensions=extensions,
        )

    @classmethod
    def storage(
        cls,
        code: StorageErrorCode,
        detail: str,
        *,
        query: str | None = None,
        table: str | None = None,
    ) -> ProblemDetail:
        """Create a storage error.

        Parameters
        ----------
        code
            Storage error code.
        detail
            Error detail message.
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
            extensions["query"] = cls._truncate(query, 200)
        if table:
            extensions["table"] = table

        return ProblemDetail(
            type=make_error_type(ErrorCategory.STORAGE, code.value),
            title="Storage Error",
            detail=detail,
            status=_STATUS_CODES.get(code.value, 500),
            extensions=extensions if extensions else {},
        )

    @classmethod
    def not_found(
        cls,
        resource_type: str,
        identifier: str,
        *,
        suggestion: str | None = None,
    ) -> ProblemDetail:
        """Create a resource not found error.

        Parameters
        ----------
        resource_type
            Type of resource (e.g., "operation", "job", "dataset").
        identifier
            Resource identifier.
        suggestion
            Suggestion for finding the resource.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        extensions: dict[str, Any] = {
            "resource_type": resource_type,
            "resource_id": identifier,
        }
        if suggestion:
            extensions["suggestion"] = suggestion

        return ProblemDetail(
            type=NOT_FOUND.type_uri,
            title=f"{resource_type.title()} Not Found",
            detail=f"{resource_type.title()} not found: {identifier}",
            status=404,
            extensions=extensions,
        )

    @classmethod
    def internal(
        cls,
        message: str,
        *,
        cause: Exception | None = None,
        operation_id: str | None = None,
        debug: bool = False,
    ) -> ProblemDetail:
        """Create an internal error.

        Parameters
        ----------
        message
            Error message.
        cause
            Underlying exception.
        operation_id
            Operation context.
        debug
            Include debug information.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        extensions: dict[str, Any] = {}
        if operation_id:
            extensions["operation_id"] = operation_id
        if cause is not None:
            extensions["cause_type"] = type(cause).__name__
            if debug:
                extensions["cause_message"] = str(cause)
                extensions["traceback"] = traceback.format_exc()

        detail = message if debug else "An unexpected error occurred"

        return ProblemDetail(
            type=INTERNAL_ERROR.type_uri,
            title="Internal Error",
            detail=detail,
            status=500,
            instance=f"/operations/{operation_id}" if operation_id else None,
            extensions=extensions if extensions else {},
        )

    @classmethod
    def config(
        cls,
        code: ConfigErrorCode,
        detail: str,
        *,
        path: str | None = None,
        key: str | None = None,
    ) -> ProblemDetail:
        """Create a configuration error.

        Parameters
        ----------
        code
            Config error code.
        detail
            Error detail message.
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
            detail=detail,
            status=_STATUS_CODES.get(code.value, 400),
            extensions=extensions if extensions else {},
        )

    @classmethod
    def service(
        cls,
        code: ServiceErrorCode,
        service: str,
        detail: str,
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
        detail
            Error detail message.
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
            detail=detail,
            status=_STATUS_CODES.get(code.value, 503),
            extensions=extensions,
        )

    @classmethod
    def job(
        cls,
        code: JobErrorCode,
        job_id: str,
        detail: str,
    ) -> ProblemDetail:
        """Create a job error.

        Parameters
        ----------
        code
            Job error code.
        job_id
            Job identifier.
        detail
            Error detail message.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        title = "Job Not Found" if code is JobErrorCode.NOT_FOUND else "Job Error"
        return ProblemDetail(
            type=make_error_type(ErrorCategory.JOB, code.value),
            title=title,
            detail=detail,
            status=_STATUS_CODES.get(code.value, 500),
            extensions={"job_id": job_id},
        )

    @classmethod
    def domain(
        cls,
        domain: str,
        code: str,
        title: str,
        detail: str,
        *,
        status: int = 400,
        **extensions: object,
    ) -> ProblemDetail:
        """Create a domain-specific error.

        Use for errors that don't fit standard categories.

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
        ProblemDetail
            Structured error.
        """
        ext: dict[str, Any] = dict(extensions) if extensions else {}

        return ProblemDetail(
            type=f"urn:codeintel:{domain}:{code}",
            title=title,
            detail=detail,
            status=status,
            extensions=ext if ext else {},
        )

    @classmethod
    def from_error_code(
        cls,
        error_code: ErrorCode,
        detail: str,
        *,
        suggestion: str | None = None,
        **extensions: object,
    ) -> ProblemDetail:
        """Create error from an ErrorCode.

        Parameters
        ----------
        error_code
            Structured error code.
        detail
            Error detail message.
        suggestion
            Suggested fix.
        **extensions
            Additional context.

        Returns
        -------
        ProblemDetail
            Structured error.
        """
        ext: dict[str, Any] = dict(extensions) if extensions else {}
        if suggestion:
            ext["suggestion"] = suggestion

        return ProblemDetail(
            type=error_code.type_uri,
            title=error_code.title,
            detail=detail,
            status=error_code.status,
            extensions=ext if ext else {},
        )

    @staticmethod
    def _truncate(value: str, max_length: int) -> str:
        """Truncate string for safety.

        Parameters
        ----------
        value
            String to truncate.
        max_length
            Maximum length.

        Returns
        -------
        str
            Truncated string.
        """
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


__all__ = [
    "ProblemBuilder",
]
