"""FastMCP middleware for canonical structured error responses.

This module converts arbitrary exceptions raised by tools/resources/prompts into:

- protocol-level `mcp.McpError(mcp.types.ErrorData(..., data=<ErrorResponse>))`
- stable machine codes and safe details from `codeintel.serving.errors`
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from fastmcp.server.middleware.middleware import Middleware
from mcp import McpError
from mcp.types import ErrorData
from pydantic import ValidationError

from codeintel.serving.errors import (
    CodeIntelDomainError,
    ErrorKind,
    build_error_context_from_mcp_context,
    exception_to_error_response,
)
from codeintel.serving.errors.transport import problem_detail_from_error_response_with_context

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
            err_context = build_error_context_from_mcp_context(context)
            error_response = exception_to_error_response(exc, context=err_context)
            kind = error_response.error.kind
            jsonrpc_code = _KIND_TO_JSONRPC_CODE.get(kind, -32603)
            problem_detail = problem_detail_from_error_response_with_context(
                error_response,
                context=err_context,
            )

            self._logger.exception(
                "MCP operation failed: method=%s tool=%s resource=%s code=%s",
                err_context.operation,
                err_context.tool_name,
                err_context.resource_uri,
                error_response.error.code,
            )

            raise McpError(
                ErrorData(
                    code=jsonrpc_code,
                    message=error_response.error.message,
                    data=problem_detail.to_dict(),
                )
            ) from exc


__all__ = ["CodeIntelErrorMappingMiddleware"]
