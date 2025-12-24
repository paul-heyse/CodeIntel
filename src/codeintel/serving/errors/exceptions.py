"""Domain exception types for serving.

These exceptions are transport-agnostic and map to stable codes in the serving
error catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codeintel.serving.errors.mapping import error_from_code
from codeintel.serving.errors.models import ErrorContext, ErrorResponse


@dataclass
class CodeIntelDomainError(Exception):
    """Domain error that maps to a stable error code."""

    code: str
    params: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_error_response(self, *, context: ErrorContext) -> ErrorResponse:
        """Convert the domain error into a transport-agnostic response payload.

        Returns
        -------
        ErrorResponse
            Canonical error response for transport adapters.
        """
        return error_from_code(
            self.code,
            context=context,
            params=self.params,
            details=self.details,
        )


class SemanticViewNotFoundError(CodeIntelDomainError):
    """Raised when a semantic view identifier is unknown."""

    def __init__(self, view_id: str) -> None:
        super().__init__(code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND", params={"view_id": view_id})


class SemanticColumnNotFoundError(CodeIntelDomainError):
    """Raised when a semantic column identifier is unknown."""

    def __init__(self, view_id: str, column: str) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
            params={"view_id": view_id, "column": column},
        )


class SemanticInvalidFilterError(CodeIntelDomainError):
    """Raised when a filter predicate cannot be validated."""

    def __init__(self, *, reason: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_INVALID_FILTER",
            details={"reason": reason} if reason else {},
        )


class SemanticLimitExceededError(CodeIntelDomainError):
    """Raised when an export or query exceeds enforced limits."""

    def __init__(self, limit: int, max_limit: int) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
            params={"limit": limit, "max_limit": max_limit},
        )


class ExportNotFoundError(CodeIntelDomainError):
    """Raised when an export artifact cannot be located."""

    def __init__(self, export_id: str) -> None:
        super().__init__(code="CODEINTEL_EXPORT_NOT_FOUND", params={"export_id": export_id})


class ExportExpiredError(CodeIntelDomainError):
    """Raised when an export artifact has exceeded its TTL."""

    def __init__(self, export_id: str, *, expires_at: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_EXPIRED",
            params={"export_id": export_id},
            details={"expires_at": expires_at} if expires_at else {},
        )


class ExportCorruptError(CodeIntelDomainError):
    """Raised when an export artifact fails integrity checks."""

    def __init__(self, export_id: str) -> None:
        super().__init__(code="CODEINTEL_EXPORT_CORRUPT", params={"export_id": export_id})


class ExportTooLargeError(CodeIntelDomainError):
    """Raised when an export exceeds configured size limits."""

    def __init__(self, *, row_count: int | None = None) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_TOO_LARGE",
            details={"row_count": row_count} if row_count else {},
        )


class ServingSnapshotNotMountedError(CodeIntelDomainError):
    """Raised when no snapshot is currently mounted for serving."""

    def __init__(self) -> None:
        super().__init__(code="CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED")


class ServingDBLockedError(CodeIntelDomainError):
    """Raised when the serving DuckDB instance is locked."""

    def __init__(self) -> None:
        super().__init__(code="CODEINTEL_SERVING_DB_LOCKED")


class AuthForbiddenError(CodeIntelDomainError):
    """Raised when a request is denied due to authorization failures."""

    def __init__(self, *, reason: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_AUTH_FORBIDDEN",
            details={"reason": reason} if reason else {},
        )


class MetaArtifactNotFoundError(CodeIntelDomainError):
    """Raised when a requested meta artifact is missing from storage."""

    def __init__(self, artifact: str) -> None:
        super().__init__(code="CODEINTEL_META_ARTIFACT_NOT_FOUND", params={"artifact": artifact})


class MetaSqlUnsafeError(CodeIntelDomainError):
    """Raised when SQL provided to meta endpoints fails safety checks."""

    def __init__(self, view_id: str) -> None:
        super().__init__(code="CODEINTEL_META_SQL_UNSAFE", params={"view_id": view_id})


class SearchIndexMissingError(CodeIntelDomainError):
    """Raised when the serving search index is missing."""

    def __init__(self) -> None:
        super().__init__(code="CODEINTEL_SEARCH_INDEX_MISSING")


class LineageMetadataMissingError(CodeIntelDomainError):
    """Raised when derived lineage metadata is missing."""

    def __init__(self, *, table: str = "metadata.derived_lineage_columns") -> None:
        super().__init__(code="CODEINTEL_LINEAGE_MISSING", params={"table": table})


__all__ = [
    "AuthForbiddenError",
    "CodeIntelDomainError",
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
]
