"""Transport-agnostic error catalog and helpers for serving."""

from codeintel.serving.errors.catalog import ERROR_CODE_CATALOG
from codeintel.serving.errors.exceptions import (
    AuthForbiddenError,
    CodeIntelDomainError,
    ExportCorruptError,
    ExportExpiredError,
    ExportNotFoundError,
    ExportTooLargeError,
    LineageMetadataMissingError,
    MetaArtifactNotFoundError,
    MetaSqlUnsafeError,
    SearchIndexMissingError,
    SemanticColumnNotFoundError,
    SemanticInvalidFilterError,
    SemanticLimitExceededError,
    SemanticViewNotFoundError,
    ServingDBLockedError,
    ServingSnapshotNotMountedError,
)
from codeintel.serving.errors.mapping import (
    build_error_context_from_http_request,
    build_error_context_from_mcp_context,
    exception_to_error_response,
)
from codeintel.serving.errors.models import ErrorContext, ErrorInfo, ErrorKind, ErrorResponse
from codeintel.serving.errors.templates import ErrorInfoTemplate

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
    "LineageMetadataMissingError",
    "MetaArtifactNotFoundError",
    "MetaSqlUnsafeError",
    "SearchIndexMissingError",
    "SemanticColumnNotFoundError",
    "SemanticInvalidFilterError",
    "SemanticLimitExceededError",
    "SemanticViewNotFoundError",
    "ServingDBLockedError",
    "ServingSnapshotNotMountedError",
    "build_error_context_from_http_request",
    "build_error_context_from_mcp_context",
    "exception_to_error_response",
]
