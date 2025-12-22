"""Transport-agnostic helpers for mapping exceptions into canonical errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from codeintel.core.execution.ids import new_uuid_hex, new_uuid_str
from codeintel.serving.errors.catalog import ERROR_CODE_CATALOG
from codeintel.serving.errors.models import ErrorContext, ErrorInfo, ErrorResponse
from codeintel.serving.uris import EXPORT_RESOURCE_PREFIX, META_VIEWS_SQL_URI

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import Request
    from fastmcp.server.middleware.middleware import MiddlewareContext


_CORRELATION_ID_HEADER = "X-Correlation-ID"


@dataclass(frozen=True, slots=True)
class _ParsedOperationContext:
    operation: str
    tool_name: str | None
    resource_uri: str | None
    view_id: str | None
    export_id: str | None


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
        "debug_id": context.debug_id or new_uuid_str(),
        "ts": datetime.now(UTC).isoformat(),
    }
    return {k: v for k, v in details.items() if v is not None}


def _extract_view_id_from_path(path: str) -> str | None:
    if path.startswith("/v1/semantic/views/"):
        remainder = path.removeprefix("/v1/semantic/views/")
        return remainder.split("/", 1)[0] or None
    if path.startswith("/v1/export/semantic/"):
        remainder = path.removeprefix("/v1/export/semantic/")
        return remainder.split("/", 1)[0] or None
    return None


def _http_request_id(request: Request) -> str | None:
    raw = getattr(request.state, "correlation_id", None)
    if isinstance(raw, str) and raw:
        return raw
    header = request.headers.get(_CORRELATION_ID_HEADER)
    if header:
        return header.strip() or None
    return None


def build_error_context_from_http_request(request: Request) -> ErrorContext:
    """Build error context for HTTP requests.

    Returns
    -------
    ErrorContext
        Normalized error context for the request.
    """
    operation = f"http:{request.method} {request.url.path}"
    return ErrorContext(
        operation=operation,
        request_id=_http_request_id(request),
        view_id=_extract_view_id_from_path(request.url.path),
    )


def _parse_tool_call_context(
    context: MiddlewareContext[object], method: str
) -> _ParsedOperationContext:
    tool_name: str | None = None
    view_id: str | None = None
    export_id: str | None = None

    tool_name_obj = getattr(context.message, "name", None)
    if isinstance(tool_name_obj, str):
        tool_name = tool_name_obj
    args_obj = getattr(context.message, "arguments", None)
    if isinstance(args_obj, dict):
        raw_view_id = args_obj.get("view_id")
        if raw_view_id is not None:
            view_id = str(raw_view_id)
        raw_export_id = args_obj.get("export_id")
        if raw_export_id is not None:
            export_id = str(raw_export_id)
    return _ParsedOperationContext(
        operation=method,
        tool_name=tool_name,
        resource_uri=None,
        view_id=view_id,
        export_id=export_id,
    )


def _parse_resource_read_context(
    context: MiddlewareContext[object], method: str
) -> _ParsedOperationContext:
    resource_uri: str | None = None
    export_id: str | None = None

    uri_obj = getattr(context.message, "uri", None)
    if isinstance(uri_obj, str):
        resource_uri = uri_obj
        if resource_uri.startswith(EXPORT_RESOURCE_PREFIX):
            remainder = resource_uri.removeprefix(EXPORT_RESOURCE_PREFIX)
            export_id = remainder.split("/", 1)[0].split("?", 1)[0] or None

    return _ParsedOperationContext(
        operation=method,
        tool_name=None,
        resource_uri=resource_uri,
        export_id=export_id,
        view_id=None,
    )


def _parse_operation_context(context: MiddlewareContext[object]) -> _ParsedOperationContext:
    method = context.method or "unknown"
    if method == "tools/call":
        return _parse_tool_call_context(context, method)
    if method == "resources/read":
        return _parse_resource_read_context(context, method)
    return _ParsedOperationContext(
        operation=method,
        tool_name=None,
        resource_uri=None,
        view_id=None,
        export_id=None,
    )


def build_error_context_from_mcp_context(context: MiddlewareContext[object]) -> ErrorContext:
    """Build error context for MCP middleware execution.

    Returns
    -------
    ErrorContext
        Normalized error context for the MCP request.
    """
    parsed = _parse_operation_context(context)
    fastmcp_ctx = context.fastmcp_context
    request_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None

    if fastmcp_ctx is not None:
        try:
            session_id_obj = getattr(fastmcp_ctx, "session_id", None)
        except RuntimeError:
            session_id_obj = None
        if isinstance(session_id_obj, str) and session_id_obj:
            request_id = session_id_obj

        try:
            snapshot_obj = getattr(fastmcp_ctx, "snapshot", None)
        except RuntimeError:
            snapshot_obj = None
        if isinstance(snapshot_obj, dict):
            repo_obj = snapshot_obj.get("repo")
            commit_obj = snapshot_obj.get("commit")
            run_id_obj = snapshot_obj.get("run_id")
            if repo_obj is not None:
                repo = str(repo_obj)
            if commit_obj is not None:
                commit = str(commit_obj)
            if run_id_obj is not None:
                run_id = str(run_id_obj)

    if request_id is None:
        request_id = new_uuid_hex()

    return ErrorContext(
        operation=parsed.operation,
        tool_name=parsed.tool_name,
        resource_uri=parsed.resource_uri,
        view_id=parsed.view_id,
        export_id=parsed.export_id,
        repo=repo,
        commit=commit,
        run_id=run_id,
        request_id=request_id,
    )


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
        and context.resource_uri.startswith(EXPORT_RESOURCE_PREFIX)
    )
    is_meta_views_sql = isinstance(context.resource_uri, str) and context.resource_uri.startswith(
        META_VIEWS_SQL_URI
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


__all__ = [
    "build_error_context_from_http_request",
    "build_error_context_from_mcp_context",
    "error_from_code",
    "exception_to_error_response",
]
