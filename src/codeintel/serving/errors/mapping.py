"""Transport-agnostic helpers for mapping exceptions into canonical errors."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import ValidationError

from codeintel.serving.errors.catalog import ERROR_CODE_CATALOG
from codeintel.serving.errors.models import ErrorContext, ErrorInfo, ErrorResponse

if TYPE_CHECKING:
    from collections.abc import Mapping


def _context_to_details(context: ErrorContext | None) -> dict[str, Any]:
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


def error_from_code(
    code: str,
    *,
    context: ErrorContext | None = None,
    params: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> ErrorResponse:
    """Render an error response from a catalog code and optional context.

    Returns
    -------
    ErrorResponse
        Serialized error payload for clients.
    """
    tmpl = ERROR_CODE_CATALOG.get(code)
    if tmpl is None:
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


def exception_to_error_response(exc: Exception, *, context: ErrorContext) -> ErrorResponse:
    """Map an arbitrary exception to a canonical error response.

    Returns
    -------
    ErrorResponse
        Canonical error response derived from the exception.
    """
    domain_mapper = getattr(exc, "to_error_response", None)
    if callable(domain_mapper):
        mapped = domain_mapper(context=context)
        if isinstance(mapped, ErrorResponse):
            return mapped

    is_export = context.tool_name == "semantic_export" or (
        isinstance(context.resource_uri, str)
        and context.resource_uri.startswith("codeintel://exports/")
    )
    is_meta_views_sql = isinstance(context.resource_uri, str) and context.resource_uri.startswith(
        "codeintel://meta/views_sql"
    )

    code: str
    params: dict[str, Any] | None = None
    details: dict[str, Any] | None = None

    if isinstance(exc, ValidationError):
        code = (
            "CODEINTEL_EXPORT_INVALID_REQUEST" if is_export else "CODEINTEL_SEMANTIC_INVALID_QUERY"
        )
        details = {"validation_errors": exc.errors()[:10]}
    elif isinstance(exc, TimeoutError):
        code = "CODEINTEL_EXPORT_UNAVAILABLE" if is_export else "CODEINTEL_SEMANTIC_QUERY_TIMEOUT"
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
        code = (
            "CODEINTEL_EXPORT_INTERNAL_ERROR" if is_export else "CODEINTEL_SEMANTIC_INTERNAL_ERROR"
        )
        details = {"exception_type": type(exc).__name__}

    return error_from_code(code, context=context, params=params, details=details)


__all__ = ["error_from_code", "exception_to_error_response"]
