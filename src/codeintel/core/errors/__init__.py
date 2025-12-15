"""Unified error taxonomy and exception types for CodeIntel.

This module consolidates error definitions from across the codebase into
a single, canonical source of truth. It provides:

- RFC 9457 Problem Details support
- Structured error codes with categories
- Base exception classes for domain-specific errors
- Factory functions for common error patterns

Import Examples
---------------
>>> # Base types and protocols
>>> from codeintel.core.errors import CodeIntelError, ProblemDetail, ErrorCode

>>> # Error codes
>>> from codeintel.core.errors import INTERNAL_ERROR, NOT_FOUND, QUERY_FAILED

>>> # Factory functions
>>> from codeintel.core.errors import make_error_type

>>> # Category enums
>>> from codeintel.core.errors import ErrorCategory, ValidationErrorCode
"""

from __future__ import annotations

# Base exception classes
from codeintel.core.errors.base import (
    CodeIntelError,
    CodeIntelOperationError,
    CodeIntelStorageError,
    CodeIntelValidationError,
    ErrorContext,
    aggregate_errors,
)

# Execution errors
from codeintel.core.errors.execution import (
    PLUGIN_CATCHABLE_ERRORS,
    PluginFatalError,
    PluginSkippedError,
    PluginSkipRequestError,
    PluginTimeoutError,
)

# Problem Details
from codeintel.core.errors.problem_details import (
    ProblemDetail,
    ProblemDetailBuilder,
    generate_instance_id,
)

# Schema errors
from codeintel.core.errors.schema import (
    SCHEMA_DIGEST_FAILED,
    SCHEMA_LOAD_FAILED,
    SCHEMA_NOT_FOUND,
    SCHEMA_VALIDATION_FAILED,
    SchemaDigestError,
    SchemaError,
    SchemaErrorCode,
    SchemaLoadError,
    SchemaNotFoundError,
    SchemaValidationError,
)

# Storage errors
from codeintel.core.errors.storage import (
    ColumnNotFoundError,
    QueryError,
    StorageColumnNotFoundError,
    StorageQueryError,
    StorageTableNotFoundError,
    TableNotFoundError,
)

# Taxonomy - Categories and Codes
from codeintel.core.errors.taxonomy import (
    # Operation codes
    ALREADY_EXISTS,
    # Service codes
    AUTH_FAILED,
    CANCELLED,
    # Storage codes
    COLUMN_NOT_FOUND,
    # Config codes
    CONFIG_FILE_NOT_FOUND,
    CONFIG_INVALID_VALUE,
    CONFIG_PARSE_ERROR,
    CONFIG_SCHEMA_VIOLATION,
    CONNECTION_FAILED,
    # Validation codes
    CONSTRAINT_VIOLATION,
    CORRUPTION_DETECTED,
    DEPENDENCY_FAILED,
    INTERNAL_ERROR,
    INVALID_FORMAT,
    INVALID_TYPE,
    # Job codes
    JOB_ALREADY_RUNNING,
    JOB_EXPIRED,
    JOB_FAILED,
    JOB_NOT_FOUND,
    MISSING_REQUIRED,
    NOT_FOUND,
    OUT_OF_RANGE,
    PERMISSION_DENIED,
    # Plugin codes
    PLUGIN_FATAL,
    PLUGIN_SKIP_REQUEST,
    PLUGIN_SKIPPED,
    PLUGIN_TIMEOUT,
    QUERY_FAILED,
    RATE_LIMITED,
    SCHEMA_MISMATCH,
    SERVICE_UNAVAILABLE,
    # Utilities
    STATUS_CODES,
    TABLE_NOT_FOUND,
    TIMEOUT,
    # Code enums
    ConfigErrorCode,
    # Categories
    ErrorCategory,
    # ErrorCode class
    ErrorCode,
    JobErrorCode,
    OperationErrorCode,
    PluginErrorCode,
    ServiceErrorCode,
    StorageErrorCode,
    ValidationErrorCode,
    make_error_type,
)

__all__ = [
    # Operation codes
    "ALREADY_EXISTS",
    # Service codes
    "AUTH_FAILED",
    "CANCELLED",
    # Storage codes
    "COLUMN_NOT_FOUND",
    # Config codes
    "CONFIG_FILE_NOT_FOUND",
    "CONFIG_INVALID_VALUE",
    "CONFIG_PARSE_ERROR",
    "CONFIG_SCHEMA_VIOLATION",
    "CONNECTION_FAILED",
    # Validation codes
    "CONSTRAINT_VIOLATION",
    "CORRUPTION_DETECTED",
    "DEPENDENCY_FAILED",
    "INTERNAL_ERROR",
    "INVALID_FORMAT",
    "INVALID_TYPE",
    # Job codes
    "JOB_ALREADY_RUNNING",
    "JOB_EXPIRED",
    "JOB_FAILED",
    "JOB_NOT_FOUND",
    "MISSING_REQUIRED",
    "NOT_FOUND",
    "OUT_OF_RANGE",
    "PERMISSION_DENIED",
    # Problem Details
    # Execution errors
    "PLUGIN_CATCHABLE_ERRORS",
    # Plugin codes
    "PLUGIN_FATAL",
    "PLUGIN_SKIPPED",
    "PLUGIN_SKIP_REQUEST",
    "PLUGIN_TIMEOUT",
    "QUERY_FAILED",
    "RATE_LIMITED",
    "SCHEMA_MISMATCH",
    "SERVICE_UNAVAILABLE",
    # Schema codes and errors
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
    # Utilities
    "STATUS_CODES",
    "TABLE_NOT_FOUND",
    "TIMEOUT",
    # Base exception classes
    "CodeIntelError",
    "CodeIntelOperationError",
    "CodeIntelStorageError",
    "CodeIntelValidationError",
    "ColumnNotFoundError",
    # Code enums
    "ConfigErrorCode",
    # Categories
    "ErrorCategory",
    # ErrorCode class
    "ErrorCode",
    "ErrorContext",
    "JobErrorCode",
    "OperationErrorCode",
    "PluginErrorCode",
    "PluginFatalError",
    "PluginSkipRequestError",
    "PluginSkippedError",
    "PluginTimeoutError",
    "ProblemDetail",
    "ProblemDetailBuilder",
    # Storage errors
    "QueryError",
    "ServiceErrorCode",
    "StorageColumnNotFoundError",
    "StorageErrorCode",
    "StorageQueryError",
    "StorageTableNotFoundError",
    "TableNotFoundError",
    "ValidationErrorCode",
    "aggregate_errors",
    "generate_instance_id",
    "make_error_type",
]
