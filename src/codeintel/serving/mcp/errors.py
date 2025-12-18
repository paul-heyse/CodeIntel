"""Compatibility shim for serving error models.

The canonical serving error catalog is transport-agnostic and lives under
`codeintel.serving.errors`.

This module remains as a thin re-export surface for historical imports under
`codeintel.serving.mcp.*`.
"""

from codeintel.serving.errors import (
    ERROR_CODE_CATALOG,
    AuthForbiddenError,
    CodeIntelDomainError,
    ErrorContext,
    ErrorInfo,
    ErrorInfoTemplate,
    ErrorKind,
    ErrorResponse,
    ExportCorruptError,
    ExportExpiredError,
    ExportNotFoundError,
    ExportTooLargeError,
    MetaArtifactNotFoundError,
    MetaSqlUnsafeError,
    SemanticColumnNotFoundError,
    SemanticInvalidFilterError,
    SemanticLimitExceededError,
    SemanticViewNotFoundError,
    ServingDBLockedError,
    ServingSnapshotNotMountedError,
    exception_to_error_response,
)
from codeintel.serving.errors.mapping import error_from_code

__all__ = [
    "ERROR_CODE_CATALOG",
    "AuthForbiddenError",
    "CodeIntelDomainError",
    "ErrorContext",
    "ErrorInfo",
    "ErrorInfoTemplate",
    "ErrorKind",
    "ErrorResponse",
    "ExportCorruptError",
    "ExportExpiredError",
    "ExportNotFoundError",
    "ExportTooLargeError",
    "MetaArtifactNotFoundError",
    "MetaSqlUnsafeError",
    "SemanticColumnNotFoundError",
    "SemanticInvalidFilterError",
    "SemanticLimitExceededError",
    "SemanticViewNotFoundError",
    "ServingDBLockedError",
    "ServingSnapshotNotMountedError",
    "error_from_code",
    "exception_to_error_response",
]
