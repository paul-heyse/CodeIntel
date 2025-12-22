"""FastMCP middleware for OpenTelemetry spans and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp.server.middleware.middleware import Middleware

from codeintel.core.execution.ids import new_uuid_hex
from codeintel.observability import operations
from codeintel.observability.context import correlation_context

if TYPE_CHECKING:
    from fastmcp.server.middleware.middleware import CallNext, MiddlewareContext


class McpOpenTelemetryMiddleware(Middleware):
    """Create spans and metrics for MCP message handling."""

    async def on_message(
        self,
        context: MiddlewareContext[object],
        call_next: CallNext[object, object],
    ) -> object:
        """Wrap MCP message handling in an observability span.

        Parameters
        ----------
        context
            Middleware context for the MCP request.
        call_next
            Handler to invoke the next middleware or endpoint.

        Returns
        -------
        object
            Response returned by the downstream handler.
        """
        return await self._handle_message(context, call_next)

    @staticmethod
    async def _handle_message(
        context: MiddlewareContext[object],
        call_next: CallNext[object, object],
    ) -> object:
        method = context.method or "unknown"
        tool_name: str | None = None
        if method == "tools/call":
            msg = getattr(context, "message", None)
            tool_name_obj = getattr(msg, "name", None) if msg is not None else None
            if isinstance(tool_name_obj, str) and tool_name_obj:
                tool_name = tool_name_obj

        operation = f"{method}:{tool_name}" if tool_name else method

        fastmcp_ctx = context.fastmcp_context
        session_id: str | None = None
        if fastmcp_ctx is not None:
            try:
                session_id_obj = getattr(fastmcp_ctx, "session_id", None)
            except RuntimeError:
                session_id_obj = None
            if isinstance(session_id_obj, str) and session_id_obj:
                session_id = session_id_obj

        correlation_id = session_id or new_uuid_hex()

        with (
            correlation_context(correlation_id),
            operations.observe_operation(
                component="mcp",
                operation=operation,
                attributes={
                    "mcp.method": method,
                    "mcp.tool_name": tool_name or "",
                },
            ),
        ):
            return await call_next(context)


__all__ = ["McpOpenTelemetryMiddleware"]
