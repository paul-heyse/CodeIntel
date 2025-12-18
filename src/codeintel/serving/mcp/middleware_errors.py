"""FastMCP middleware for canonical structured error responses.

This module converts arbitrary exceptions raised by tools/resources/prompts into:

- protocol-level `mcp.McpError(mcp.types.ErrorData(..., data=<ErrorResponse>))`
- stable machine codes and safe details from `codeintel.serving.mcp.errors`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from fastmcp.server.middleware.middleware import Middleware
from mcp import McpError
from mcp.types import ErrorData
from pydantic import ValidationError

from codeintel.serving.mcp.errors import (
    CodeIntelDomainError,
    ErrorContext,
    ErrorKind,
    exception_to_error_response,
)

if TYPE_CHECKING:
    from fastmcp.server.middleware.middleware import CallNext, MiddlewareContext

LOG = logging.getLogger(__name__)

_KIND_TO_JSONRPC_CODE: Final[dict[ErrorKind, int]] = {
    ErrorKind.invalid_request: -32602,
    ErrorKind.not_found: -32001,
    ErrorKind.expired: -32001,
    ErrorKind.corrupt: -32001,
    ErrorKind.conflict: -32000,
    ErrorKind.unavailable: -32000,
    ErrorKind.timeout: -32000,
    ErrorKind.internal: -32603,
}


@dataclass(frozen=True, slots=True)
class _ParsedOperationContext:
    operation: str
    tool_name: str | None
    resource_uri: str | None
    view_id: str | None
    export_id: str | None


def _parse_tool_call_context(context: MiddlewareContext[object], method: str) -> _ParsedOperationContext:
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


def _parse_resource_read_context(context: MiddlewareContext[object], method: str) -> _ParsedOperationContext:
    resource_uri: str | None = None
    export_id: str | None = None

    uri_obj = getattr(context.message, "uri", None)
    if isinstance(uri_obj, str):
        resource_uri = uri_obj
        export_marker = "codeintel://exports/"
        if resource_uri.startswith(export_marker):
            remainder = resource_uri.removeprefix(export_marker)
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


def _build_error_context(
    context: MiddlewareContext[object],
    parsed: _ParsedOperationContext,
) -> ErrorContext:
    fastmcp_ctx = context.fastmcp_context
    request_id: str | None = None
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None

    if fastmcp_ctx is not None:
        session_id_obj = getattr(fastmcp_ctx, "session_id", None)
        if isinstance(session_id_obj, str) and session_id_obj:
            request_id = session_id_obj

        snapshot_obj = getattr(fastmcp_ctx, "snapshot", None)
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


class CodeIntelErrorMappingMiddleware(Middleware):
    """Convert exceptions into canonical protocol-level McpError payloads."""

    def __init__(self) -> None:
        self._logger = LOG

    async def on_message(
        self,
        context: MiddlewareContext[object],
        call_next: CallNext[object, object],
    ) -> object:
        """Convert known exceptions to protocol-level McpError with structured data.

        Parameters
        ----------
        context
            Middleware execution context with method name and session metadata.
        call_next
            Continuation callback to execute the underlying handler.

        Returns
        -------
        object
            Successful handler result.

        Raises
        ------
        McpError
            If the operation fails and is converted into a canonical structured MCP error.
        """
        try:
            return await call_next(context)
        except McpError:
            raise
        except (
            CodeIntelDomainError,
            FileNotFoundError,
            KeyError,
            PermissionError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValidationError,
            ValueError,
        ) as exc:
            parsed = _parse_operation_context(context)
            err_context = _build_error_context(context, parsed)
            error_response = exception_to_error_response(exc, context=err_context)
            kind = error_response.error.kind
            jsonrpc_code = _KIND_TO_JSONRPC_CODE.get(kind, -32603)

            self._logger.exception(
                "MCP operation failed: method=%s tool=%s resource=%s code=%s",
                parsed.operation,
                parsed.tool_name,
                parsed.resource_uri,
                error_response.error.code,
            )

            raise McpError(
                ErrorData(
                    code=jsonrpc_code,
                    message=error_response.error.message,
                    data=error_response.model_dump(mode="json"),
                )
            ) from exc


__all__ = ["CodeIntelErrorMappingMiddleware"]
