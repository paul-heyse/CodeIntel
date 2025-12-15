"""Schema-specific error types for explicit exception handling.

This module defines a hierarchy of exceptions for schema operations,
enabling precise error handling without blind excepts (BLE001).

The schema error taxonomy:

    urn:codeintel:schema/
    ├── not-found          # Schema not found for table key
    ├── load-failed        # Schema failed to load from source
    ├── validation-failed  # Schema validation failed
    └── digest-failed      # Schema digest computation failed

Examples
--------
>>> from codeintel.core.errors.schema import SchemaNotFoundError
>>> raise SchemaNotFoundError("analytics.function_metrics")
Traceback (most recent call last):
    ...
SchemaNotFoundError: No schema found for table key: analytics.function_metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from codeintel.core.errors.base import CodeIntelError
from codeintel.core.errors.taxonomy import ErrorCategory, ErrorCode


class SchemaErrorCode(Enum):
    """Schema-specific error codes."""

    NOT_FOUND = "not-found"
    LOAD_FAILED = "load-failed"
    VALIDATION_FAILED = "validation-failed"
    DIGEST_FAILED = "digest-failed"


# -----------------------------------------------------------------------------
# Schema Error Codes
# -----------------------------------------------------------------------------

SCHEMA_NOT_FOUND = ErrorCode(
    ErrorCategory.OPERATION,
    "schema/not-found",
    404,
    "Schema Not Found",
)

SCHEMA_LOAD_FAILED = ErrorCode(
    ErrorCategory.OPERATION,
    "schema/load-failed",
    500,
    "Schema Load Failed",
)

SCHEMA_VALIDATION_FAILED = ErrorCode(
    ErrorCategory.VALIDATION,
    "schema/validation-failed",
    400,
    "Schema Validation Failed",
)

SCHEMA_DIGEST_FAILED = ErrorCode(
    ErrorCategory.OPERATION,
    "schema/digest-failed",
    500,
    "Schema Digest Failed",
)


# -----------------------------------------------------------------------------
# Schema Exception Classes
# -----------------------------------------------------------------------------


@dataclass
class SchemaError(CodeIntelError):
    """Base exception for all schema-related errors.

    Attributes
    ----------
    table_key
        The table key associated with this schema error.

    Examples
    --------
    >>> error = SchemaError(
    ...     error_code=SCHEMA_NOT_FOUND,
    ...     detail="Schema missing",
    ...     table_key="analytics.function_metrics",
    ... )
    >>> error.table_key
    'analytics.function_metrics'
    """

    table_key: str | None = None

    def __post_init__(self) -> None:
        """Initialize with table_key in extensions."""
        if self.table_key and "table_key" not in self.extensions:
            self.extensions["table_key"] = self.table_key
        super().__post_init__()


class SchemaNotFoundError(SchemaError):
    """Raise when a schema is not found for a table key.

    Examples
    --------
    >>> raise SchemaNotFoundError("analytics.function_metrics")
    Traceback (most recent call last):
        ...
    SchemaNotFoundError: No schema found for table key: analytics.function_metrics
    """

    def __init__(self, table_key: str) -> None:
        """Initialize with the missing table key.

        Parameters
        ----------
        table_key
            The table key that was not found.
        """
        super().__init__(
            error_code=SCHEMA_NOT_FOUND,
            detail=f"No schema found for table key: {table_key}",
            table_key=table_key,
        )


class SchemaLoadError(SchemaError):
    """Raise when a schema cannot be loaded from its source.

    Examples
    --------
    >>> try:
    ...     raise ValueError("file not found")
    ... except ValueError as e:
    ...     raise SchemaLoadError("analytics.function_metrics", e) from e
    """

    def __init__(self, table_key: str, cause: Exception) -> None:
        """Initialize with the table key and underlying cause.

        Parameters
        ----------
        table_key
            The table key that failed to load.
        cause
            The underlying exception that caused the load failure.
        """
        super().__init__(
            error_code=SCHEMA_LOAD_FAILED,
            detail=f"Failed to load schema for {table_key}: {cause}",
            table_key=table_key,
            cause=cause,
            extensions={"cause_type": type(cause).__name__},
        )


@dataclass
class SchemaValidationError(SchemaError):
    """Raise when schema validation fails.

    Attributes
    ----------
    violations
        List of validation error messages.

    Examples
    --------
    >>> raise SchemaValidationError(
    ...     "analytics.function_metrics",
    ...     ["column 'loc' missing", "type mismatch for 'name'"],
    ... )
    """

    violations: list[str] = field(default_factory=list)

    def __init__(self, table_key: str, violations: list[str]) -> None:
        """Initialize with validation violations.

        Parameters
        ----------
        table_key
            The table key that failed validation.
        violations
            List of validation error messages.
        """
        self.violations = violations
        detail = f"Schema validation failed for {table_key}: {'; '.join(violations)}"
        super().__init__(
            error_code=SCHEMA_VALIDATION_FAILED,
            detail=detail,
            table_key=table_key,
            extensions={"violations": violations},
        )


class SchemaDigestError(SchemaError):
    """Raise when schema digest computation fails.

    Examples
    --------
    >>> try:
    ...     raise ValueError("invalid JSON")
    ... except ValueError as e:
    ...     raise SchemaDigestError("analytics.function_metrics", e) from e
    """

    def __init__(self, table_key: str, cause: Exception) -> None:
        """Initialize with the table key and underlying cause.

        Parameters
        ----------
        table_key
            The table key for which digest computation failed.
        cause
            The underlying exception.
        """
        super().__init__(
            error_code=SCHEMA_DIGEST_FAILED,
            detail=f"Failed to compute digest for {table_key}: {cause}",
            table_key=table_key,
            cause=cause,
            extensions={"cause_type": type(cause).__name__},
        )


__all__ = [
    "SCHEMA_DIGEST_FAILED",
    "SCHEMA_LOAD_FAILED",
    "SCHEMA_NOT_FOUND",
    "SCHEMA_VALIDATION_FAILED",
    "SchemaDigestError",
    "SchemaError",
    "SchemaErrorCode",
    "SchemaLoadError",
    "SchemaNotFoundError",
    "SchemaValidationError",
]
