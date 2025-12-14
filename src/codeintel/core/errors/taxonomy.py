"""Error taxonomy and code definitions.

This module defines the canonical error hierarchy for CodeIntel:

    urn:codeintel:
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
    ├── job/
    │   ├── not-found
    │   ├── already-running
    │   ├── failed
    │   └── expired
    └── plugin/
        ├── fatal
        ├── timeout
        ├── skipped
        └── skip-request
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    """Top-level error categories."""

    VALIDATION = "validation"
    OPERATION = "operation"
    STORAGE = "storage"
    CONFIG = "config"
    SERVICE = "service"
    JOB = "job"
    PLUGIN = "plugin"


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
    TABLE_NOT_FOUND = "table-not-found"
    COLUMN_NOT_FOUND = "column-not-found"


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


class PluginErrorCode(Enum):
    """Plugin execution error codes."""

    FATAL = "fatal"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    SKIP_REQUEST = "skip-request"


@dataclass(frozen=True)
class ErrorCode:
    """Structured error code with metadata.

    Attributes
    ----------
    category
        Error category.
    code
        Specific error code within category.
    status
        HTTP status code or exit code.
    title
        Human-readable title.

    Examples
    --------
    >>> code = ErrorCode(ErrorCategory.VALIDATION, "missing-required", 400, "Missing Required")
    >>> code.type_uri
    'urn:codeintel:validation/missing-required'
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
            Error type URI in format urn:codeintel:category/code.
        """
        return f"urn:codeintel:{self.category.value}/{self.code}"


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
    return f"urn:codeintel:{category.value}/{code}"


# -----------------------------------------------------------------------------
# Validation Error Codes
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Operation Error Codes
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Storage Error Codes
# -----------------------------------------------------------------------------

CONNECTION_FAILED = ErrorCode(
    ErrorCategory.STORAGE, "connection-failed", 503, "Storage Connection Failed"
)
QUERY_FAILED = ErrorCode(ErrorCategory.STORAGE, "query-failed", 500, "Query Failed")
SCHEMA_MISMATCH = ErrorCode(ErrorCategory.STORAGE, "schema-mismatch", 500, "Schema Mismatch")
CORRUPTION_DETECTED = ErrorCode(
    ErrorCategory.STORAGE, "corruption-detected", 500, "Data Corruption Detected"
)
TABLE_NOT_FOUND = ErrorCode(ErrorCategory.STORAGE, "table-not-found", 404, "Table Not Found")
COLUMN_NOT_FOUND = ErrorCode(ErrorCategory.STORAGE, "column-not-found", 404, "Column Not Found")

# -----------------------------------------------------------------------------
# Config Error Codes
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Service Error Codes
# -----------------------------------------------------------------------------

SERVICE_UNAVAILABLE = ErrorCode(ErrorCategory.SERVICE, "unavailable", 503, "Service Unavailable")
RATE_LIMITED = ErrorCode(ErrorCategory.SERVICE, "rate-limited", 429, "Rate Limited")
AUTH_FAILED = ErrorCode(
    ErrorCategory.SERVICE, "authentication-failed", 401, "Authentication Failed"
)
PERMISSION_DENIED = ErrorCode(ErrorCategory.SERVICE, "permission-denied", 403, "Permission Denied")

# -----------------------------------------------------------------------------
# Job Error Codes
# -----------------------------------------------------------------------------

JOB_NOT_FOUND = ErrorCode(ErrorCategory.JOB, "not-found", 404, "Job Not Found")
JOB_ALREADY_RUNNING = ErrorCode(ErrorCategory.JOB, "already-running", 409, "Job Already Running")
JOB_FAILED = ErrorCode(ErrorCategory.JOB, "failed", 500, "Job Execution Failed")
JOB_EXPIRED = ErrorCode(ErrorCategory.JOB, "expired", 410, "Job Results Expired")

# -----------------------------------------------------------------------------
# Plugin Error Codes
# -----------------------------------------------------------------------------

PLUGIN_FATAL = ErrorCode(ErrorCategory.PLUGIN, "fatal", 500, "Plugin Fatal Error")
PLUGIN_TIMEOUT = ErrorCode(ErrorCategory.PLUGIN, "timeout", 504, "Plugin Timeout")
PLUGIN_SKIPPED = ErrorCode(ErrorCategory.PLUGIN, "skipped", 200, "Plugin Skipped")
PLUGIN_SKIP_REQUEST = ErrorCode(ErrorCategory.PLUGIN, "skip-request", 200, "Plugin Skip Requested")


# Status code lookup for backward compatibility
STATUS_CODES: dict[str, int] = {
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
    StorageErrorCode.TABLE_NOT_FOUND.value: 404,
    StorageErrorCode.COLUMN_NOT_FOUND.value: 404,
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
    PluginErrorCode.FATAL.value: 500,
    PluginErrorCode.TIMEOUT.value: 504,
    PluginErrorCode.SKIPPED.value: 200,
    PluginErrorCode.SKIP_REQUEST.value: 200,
}


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
    # Plugin codes
    "PLUGIN_FATAL",
    "PLUGIN_SKIPPED",
    "PLUGIN_SKIP_REQUEST",
    "PLUGIN_TIMEOUT",
    "QUERY_FAILED",
    "RATE_LIMITED",
    "SCHEMA_MISMATCH",
    "SERVICE_UNAVAILABLE",
    # Utilities
    "STATUS_CODES",
    "TABLE_NOT_FOUND",
    "TIMEOUT",
    # Code enums
    "ConfigErrorCode",
    # Categories
    "ErrorCategory",
    # ErrorCode class
    "ErrorCode",
    "JobErrorCode",
    "OperationErrorCode",
    "PluginErrorCode",
    "ServiceErrorCode",
    "StorageErrorCode",
    "ValidationErrorCode",
    "make_error_type",
]
