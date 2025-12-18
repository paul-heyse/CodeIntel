"""Canonical error models, catalog, and helpers for MCP tools and resources.

This module provides a centralized error infrastructure for MCP surfaces:
- ErrorKind enum for error categories
- ErrorInfo/ErrorResponse Pydantic models for structured errors
- ERROR_CODE_CATALOG with 20 locked-in error codes
- error_from_code() helper for constructing errors
- exception_to_error_response() mapper for exception handling
- Domain exception classes that map to stable error codes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping


# =============================================================================
# Error Kind Enum
# =============================================================================


class ErrorKind(StrEnum):
    """Coarse error categories for MCP responses.

    These categories help LLM agents decide how to handle errors:
    - invalid_request: Fix the request and retry
    - not_found: Resource doesn't exist
    - expired: Resource existed but is no longer valid
    - corrupt: Resource exists but is damaged
    - conflict: Request conflicts with current state
    - unavailable: Temporary issue, retry later
    - timeout: Operation took too long, retry with smaller scope
    - internal: Unexpected error, check logs
    """

    invalid_request = "invalid_request"
    not_found = "not_found"
    expired = "expired"
    corrupt = "corrupt"
    conflict = "conflict"
    unavailable = "unavailable"
    timeout = "timeout"
    internal = "internal"


# =============================================================================
# Error Pydantic Models
# =============================================================================


class ErrorInfo(BaseModel):
    """Canonical error payload for MCP tools and resources.

    Parameters
    ----------
    code
        Stable machine code (e.g. CODEINTEL_EXPORT_EXPIRED).
        Never change once published.
    kind
        Coarse error category.
    message
        Short, safe human-readable description.
    retryable
        Whether client can retry safely.
    hint
        What the client/agent should do next (safe guidance).
    details
        Safe structured details (no stack traces, no internal file paths).
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description="Stable machine code. Never change once published.",
        examples=["CODEINTEL_EXPORT_EXPIRED", "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND"],
    )
    kind: ErrorKind = Field(..., description="Coarse error category.")
    message: str = Field(..., description="Short, safe human-readable description.")
    retryable: bool = Field(default=False, description="Whether client can retry safely.")
    hint: str | None = Field(
        None,
        description="What the client/agent should do next (safe guidance).",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe structured details (no stack traces).",
    )


class ErrorResponse(BaseModel):
    """Canonical top-level error response for MCP tools and resources.

    Parameters
    ----------
    status
        Always "error" for error responses.
    error
        The canonical error payload.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error: ErrorInfo


# =============================================================================
# Error Code Catalog
# =============================================================================


@dataclass(frozen=True, slots=True)
class ErrorInfoTemplate:
    """Template for an error code in the catalog.

    Parameters
    ----------
    code
        Stable error code (must match catalog key).
    kind
        Error category.
    message
        Message template (may contain {placeholders}).
    hint
        Hint template for agent guidance.
    retryable
        Whether the error is retryable.
    http_status
        Corresponding HTTP status code for HTTP surface parity.
    """

    code: str
    kind: ErrorKind
    message: str
    hint: str | None = None
    retryable: bool = False
    http_status: int | None = None

    def render_message(self, params: Mapping[str, Any] | None = None) -> str:
        """Render message template with parameters.

        Parameters
        ----------
        params
            Template parameters to substitute.

        Returns
        -------
        str
            Rendered message.
        """
        return _safe_format(self.message, params)

    def render_hint(self, params: Mapping[str, Any] | None = None) -> str | None:
        """Render hint template with parameters.

        Parameters
        ----------
        params
            Template parameters to substitute.

        Returns
        -------
        str | None
            Rendered hint or None.
        """
        if self.hint is None:
            return None
        return _safe_format(self.hint, params)


def _safe_format(template: str, params: Mapping[str, Any] | None) -> str:
    """Format template with safe fallback for missing keys.

    Parameters
    ----------
    template
        String template with {placeholders}.
    params
        Parameters to substitute.

    Returns
    -------
    str
        Formatted string (missing keys kept as {key}).
    """
    if not params:
        return template

    class SafeDict(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeDict(params))


# Canonical catalog - LOCK THIS IN
# These codes are part of the public API contract and should never change
ERROR_CODE_CATALOG: dict[str, ErrorInfoTemplate] = {
    # -------- Semantic/query layer (8) --------
    "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
        kind=ErrorKind.not_found,
        http_status=404,
        retryable=False,
        message="Semantic view '{view_id}' not found.",
        hint="Call semantic_catalog to list available views, then retry.",
    ),
    "CODEINTEL_SEMANTIC_INVALID_QUERY": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INVALID_QUERY",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Invalid semantic query request.",
        hint="Validate request fields/types and compare against semantic_describe(view_id).",
    ),
    "CODEINTEL_SEMANTIC_INVALID_FILTER": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INVALID_FILTER",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Invalid filter specification.",
        hint="Check filter operators and values against the view schema (semantic_describe).",
    ),
    "CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Column '{column}' not found in semantic view '{view_id}'.",
        hint="Use semantic_describe(view_id) to list available columns.",
    ),
    "CODEINTEL_SEMANTIC_LIMIT_EXCEEDED": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Requested limit {limit} exceeds maximum {max_limit}.",
        hint="Lower the limit or use export/preview flows for large results.",
    ),
    "CODEINTEL_SEMANTIC_QUERY_TIMEOUT": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_QUERY_TIMEOUT",
        kind=ErrorKind.timeout,
        http_status=504,
        retryable=True,
        message="Semantic query timed out.",
        hint="Retry with narrower filters and/or a smaller limit; use semantic_explain.",
    ),
    "CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_QUERY_UNAVAILABLE",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Semantic query backend is temporarily unavailable.",
        hint="Retry shortly. If this persists, check serving snapshot state (serving_meta).",
    ),
    "CODEINTEL_SEMANTIC_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_SEMANTIC_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        http_status=500,
        retryable=True,
        message="Internal error while processing semantic request.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),
    # -------- Export subsystem (6) --------
    "CODEINTEL_EXPORT_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_NOT_FOUND",
        kind=ErrorKind.not_found,
        http_status=404,
        retryable=False,
        message="Export handle '{export_id}' not found.",
        hint="Re-run semantic_query/export to generate a fresh export handle.",
    ),
    "CODEINTEL_EXPORT_EXPIRED": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_EXPIRED",
        kind=ErrorKind.expired,
        http_status=410,
        retryable=True,
        message="Export handle '{export_id}' expired.",
        hint="Re-run semantic_query/export with the same parameters for a fresh handle.",
    ),
    "CODEINTEL_EXPORT_CORRUPT": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_CORRUPT",
        kind=ErrorKind.corrupt,
        http_status=500,
        retryable=True,
        message="Export '{export_id}' is corrupt or unreadable.",
        hint="Re-run semantic_export to regenerate; if it repeats, rebuild snapshot.",
    ),
    "CODEINTEL_EXPORT_TOO_LARGE": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_TOO_LARGE",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Export request is too large (rows/bytes exceed server limits).",
        hint="Add filters and/or reduce limit; consider exporting narrower slices.",
    ),
    "CODEINTEL_EXPORT_INVALID_REQUEST": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_INVALID_REQUEST",
        kind=ErrorKind.invalid_request,
        http_status=400,
        retryable=False,
        message="Invalid export request.",
        hint="Validate view_id, export_format, limit, and filters; call semantic_describe(view_id) for schema.",
    ),
    "CODEINTEL_EXPORT_UNAVAILABLE": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_UNAVAILABLE",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Export store is temporarily unavailable.",
        hint="Retry shortly. If it persists, check disk space and export TTL settings.",
    ),
    "CODEINTEL_EXPORT_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_EXPORT_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        http_status=500,
        retryable=True,
        message="Internal error while retrieving export.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),
    "CODEINTEL_META_ARTIFACT_NOT_FOUND": ErrorInfoTemplate(
        code="CODEINTEL_META_ARTIFACT_NOT_FOUND",
        kind=ErrorKind.not_found,
        http_status=404,
        retryable=False,
        message="Meta artifact '{artifact}' not found for mounted snapshot.",
        hint="Verify the snapshot was built with serving artifacts enabled.",
    ),
    "CODEINTEL_META_SQL_UNSAFE": ErrorInfoTemplate(
        code="CODEINTEL_META_SQL_UNSAFE",
        kind=ErrorKind.corrupt,
        http_status=500,
        retryable=False,
        message="Meta SQL artifact contains unsafe SQL for view '{view_id}'.",
        hint="Rebuild the snapshot and ensure compiled SQL stays within the select-only perimeter.",
    ),
    # -------- Serving snapshot / DB (4) --------
    "CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Serving snapshot is not mounted.",
        hint="Wait for snapshot publication or restart the server; check serving_meta.",
    ),
    "CODEINTEL_SERVING_SNAPSHOT_MISMATCH": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_SNAPSHOT_MISMATCH",
        kind=ErrorKind.conflict,
        http_status=409,
        retryable=True,
        message="Requested snapshot does not match the mounted serving snapshot.",
        hint="Refresh snapshot metadata (serving_meta) and retry with current identifiers.",
    ),
    "CODEINTEL_SERVING_DB_LOCKED": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_DB_LOCKED",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Serving database is busy/locked.",
        hint="Retry shortly (backoff). Consider lowering concurrency.",
    ),
    "CODEINTEL_SERVING_DB_INTERNAL_ERROR": ErrorInfoTemplate(
        code="CODEINTEL_SERVING_DB_INTERNAL_ERROR",
        kind=ErrorKind.internal,
        http_status=500,
        retryable=True,
        message="Internal serving database error.",
        hint="Retry. If it persists, inspect server logs using debug_id.",
    ),
    # -------- Schema/spec/auth (2) --------
    "CODEINTEL_SCHEMA_MANIFEST_MISSING": ErrorInfoTemplate(
        code="CODEINTEL_SCHEMA_MANIFEST_MISSING",
        kind=ErrorKind.unavailable,
        http_status=503,
        retryable=True,
        message="Schema manifest missing for the mounted snapshot.",
        hint="Rebuild/publish the snapshot or rerun schema compilation.",
    ),
    "CODEINTEL_AUTH_FORBIDDEN": ErrorInfoTemplate(
        code="CODEINTEL_AUTH_FORBIDDEN",
        kind=ErrorKind.invalid_request,
        http_status=403,
        retryable=False,
        message="Forbidden.",
        hint="This operation is not permitted by server policy/settings.",
    ),
}


# =============================================================================
# Error Context
# =============================================================================


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Safe, structured context for error mapping.

    Contains only safe identifiers for debugging - no stack traces,
    file paths, or secrets.

    Parameters
    ----------
    operation
        The operation being performed (e.g. "semantic_query").
    tool_name
        MCP tool name if applicable.
    resource_uri
        MCP resource URI if applicable.
    view_id
        Semantic view ID if applicable.
    export_id
        Export handle ID if applicable.
    repo
        Repository identifier.
    commit
        Git commit hash.
    run_id
        Build run identifier.
    request_id
        Correlation/request ID.
    debug_id
        Unique debug ID for log correlation.
    """

    operation: str
    tool_name: str | None = None
    resource_uri: str | None = None
    view_id: str | None = None
    export_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    debug_id: str | None = None


def _context_to_details(context: ErrorContext | None) -> dict[str, Any]:
    """Extract safe details from ErrorContext.

    Parameters
    ----------
    context
        Error context to extract details from.

    Returns
    -------
    dict[str, Any]
        Safe details dictionary with None values removed.
    """
    if context is None:
        return {"ts": datetime.now(UTC).isoformat()}

    details: dict[str, Any] = {
        "operation": context.operation,
        "tool_name": context.tool_name,
        "resource_uri": context.resource_uri,
        "view_id": context.view_id,
        "export_id": context.export_id,
        "repo": context.repo,
        "commit": context.commit,
        "run_id": context.run_id,
        "request_id": context.request_id,
        "debug_id": context.debug_id or str(uuid4()),
        "ts": datetime.now(UTC).isoformat(),
    }
    return {k: v for k, v in details.items() if v is not None}


# =============================================================================
# Helper Functions
# =============================================================================


def error_from_code(
    code: str,
    *,
    context: ErrorContext | None = None,
    params: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> ErrorResponse:
    """Construct ErrorResponse from catalog code.

    Parameters
    ----------
    code
        Error code from ERROR_CODE_CATALOG.
    context
        Error context for debugging.
    params
        Template parameters for message/hint.
    details
        Additional safe details to include.

    Returns
    -------
    ErrorResponse
        Canonical error response.
    """
    tmpl = ERROR_CODE_CATALOG.get(code)
    if tmpl is None:
        # Fall back to internal error if code not found
        tmpl = ERROR_CODE_CATALOG["CODEINTEL_SEMANTIC_INTERNAL_ERROR"]

    base_details = _context_to_details(context)
    if details:
        base_details.update({k: v for k, v in dict(details).items() if v is not None})

    return ErrorResponse(
        error=ErrorInfo(
            code=tmpl.code,
            kind=tmpl.kind,
            message=tmpl.render_message(params),
            retryable=tmpl.retryable,
            hint=tmpl.render_hint(params),
            details=base_details,
        )
    )


def exception_to_error_response(
    exc: Exception,
    *,
    context: ErrorContext,
) -> ErrorResponse:
    """Convert any exception to canonical ErrorResponse.

    Parameters
    ----------
    exc
        Exception to convert.
    context
        Error context for debugging.

    Returns
    -------
    ErrorResponse
        Canonical error response.
    """
    if isinstance(exc, CodeIntelDomainError):
        return exc.to_error_response(context=context)

    is_export = context.tool_name == "semantic_export" or (
        isinstance(context.resource_uri, str) and context.resource_uri.startswith("codeintel://exports/")
    )
    is_meta_views_sql = isinstance(context.resource_uri, str) and context.resource_uri.startswith(
        "codeintel://meta/views_sql"
    )

    code: str
    params: dict[str, Any] | None = None
    details: dict[str, Any] | None = None

    if isinstance(exc, ValidationError):
        code = "CODEINTEL_EXPORT_INVALID_REQUEST" if is_export else "CODEINTEL_SEMANTIC_INVALID_QUERY"
        details = {"validation_errors": exc.errors()[:10]}
    elif isinstance(exc, TimeoutError):
        if is_export:
            code = "CODEINTEL_EXPORT_UNAVAILABLE"
        else:
            code = "CODEINTEL_SEMANTIC_QUERY_TIMEOUT"
    elif isinstance(exc, KeyError) and context.view_id is not None:
        code = "CODEINTEL_SEMANTIC_VIEW_NOT_FOUND"
        params = {"view_id": context.view_id}
    elif isinstance(exc, (TypeError, ValueError)):
        if is_meta_views_sql and context.view_id is not None:
            code = "CODEINTEL_META_SQL_UNSAFE"
            params = {"view_id": context.view_id}
        elif is_export:
            code = "CODEINTEL_EXPORT_INVALID_REQUEST"
        else:
            code = "CODEINTEL_SEMANTIC_INVALID_QUERY"
    else:
        code = "CODEINTEL_EXPORT_INTERNAL_ERROR" if is_export else "CODEINTEL_SEMANTIC_INTERNAL_ERROR"
        details = {"exception_type": type(exc).__name__}

    return error_from_code(code, context=context, params=params, details=details)


# =============================================================================
# Domain Exceptions
# =============================================================================


@dataclass
class CodeIntelDomainError(Exception):
    """Domain error that maps to a stable error code.

    Domain exceptions carry their error code and parameters so they can
    be converted to ErrorResponse via exception_to_error_response().

    Parameters
    ----------
    code
        Error code from ERROR_CODE_CATALOG.
    params
        Template parameters for message/hint.
    details
        Additional safe details.
    """

    code: str
    params: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_error_response(self, *, context: ErrorContext) -> ErrorResponse:
        """Convert to canonical ErrorResponse.

        Parameters
        ----------
        context
            Error context for debugging.

        Returns
        -------
        ErrorResponse
            Canonical error response.
        """
        return error_from_code(
            self.code,
            context=context,
            params=self.params,
            details=self.details,
        )


# Specific domain exceptions for common cases


class SemanticViewNotFoundError(CodeIntelDomainError):
    """Semantic view not found in registry."""

    def __init__(self, view_id: str) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_VIEW_NOT_FOUND",
            params={"view_id": view_id},
        )


class SemanticColumnNotFoundError(CodeIntelDomainError):
    """Column not found in semantic view."""

    def __init__(self, view_id: str, column: str) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_COLUMN_NOT_FOUND",
            params={"view_id": view_id, "column": column},
        )


class SemanticInvalidFilterError(CodeIntelDomainError):
    """Invalid filter specification."""

    def __init__(self, *, reason: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_INVALID_FILTER",
            details={"reason": reason} if reason else {},
        )


class SemanticLimitExceededError(CodeIntelDomainError):
    """Requested limit exceeds maximum."""

    def __init__(self, limit: int, max_limit: int) -> None:
        super().__init__(
            code="CODEINTEL_SEMANTIC_LIMIT_EXCEEDED",
            params={"limit": limit, "max_limit": max_limit},
        )


class ExportNotFoundError(CodeIntelDomainError):
    """Export handle not found."""

    def __init__(self, export_id: str) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_NOT_FOUND",
            params={"export_id": export_id},
        )


class ExportExpiredError(CodeIntelDomainError):
    """Export handle has expired."""

    def __init__(self, export_id: str, *, expires_at: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_EXPIRED",
            params={"export_id": export_id},
            details={"expires_at": expires_at} if expires_at else {},
        )


class ExportCorruptError(CodeIntelDomainError):
    """Export data is corrupt or unreadable."""

    def __init__(self, export_id: str) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_CORRUPT",
            params={"export_id": export_id},
        )


class ExportTooLargeError(CodeIntelDomainError):
    """Export request exceeds server limits."""

    def __init__(self, *, row_count: int | None = None) -> None:
        super().__init__(
            code="CODEINTEL_EXPORT_TOO_LARGE",
            details={"row_count": row_count} if row_count else {},
        )


class ServingSnapshotNotMountedError(CodeIntelDomainError):
    """Serving snapshot is not mounted."""

    def __init__(self) -> None:
        super().__init__(code="CODEINTEL_SERVING_SNAPSHOT_NOT_MOUNTED")


class ServingDBLockedError(CodeIntelDomainError):
    """Serving database is busy/locked."""

    def __init__(self) -> None:
        super().__init__(code="CODEINTEL_SERVING_DB_LOCKED")


class AuthForbiddenError(CodeIntelDomainError):
    """Operation not permitted by server policy."""

    def __init__(self, *, reason: str | None = None) -> None:
        super().__init__(
            code="CODEINTEL_AUTH_FORBIDDEN",
            details={"reason": reason} if reason else {},
        )


class MetaArtifactNotFoundError(CodeIntelDomainError):
    """Serving meta artifact missing from the mounted snapshot."""

    def __init__(self, artifact: str) -> None:
        super().__init__(
            code="CODEINTEL_META_ARTIFACT_NOT_FOUND",
            params={"artifact": artifact},
        )


class MetaSqlUnsafeError(CodeIntelDomainError):
    """Serving meta SQL artifact violates the select-only perimeter."""

    def __init__(self, view_id: str) -> None:
        super().__init__(
            code="CODEINTEL_META_SQL_UNSAFE",
            params={"view_id": view_id},
        )


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
