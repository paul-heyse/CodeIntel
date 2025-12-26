"""FastMCP middleware for OpenTelemetry spans and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp.server.middleware.middleware import Middleware

from codeintel.core.execution.ids import new_uuid_hex
from codeintel.observability.operation_scope import observe_operation
from codeintel.observability.runtime import get_observability
from codeintel.observability.semconv import mcp_span_attributes
from codeintel.observability.semconv_keys import MCP_TOOL_NAME
from codeintel.observability.telemetry_context import telemetry_context

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

        policy = get_observability().policy
        mcp_attrs = mcp_span_attributes(
            method=method,
            tool_name=tool_name,
            policy=policy,
        )
        normalized_tool = mcp_attrs.get(MCP_TOOL_NAME)
        operation = (
            f"{method}:{normalized_tool}"
            if isinstance(normalized_tool, str) and normalized_tool
            else method
        )

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
            telemetry_context(correlation_id=correlation_id),
            observe_operation(
                component="mcp",
                operation=operation,
                attributes={
                    **mcp_attrs,
                },
            ),
        ):
            return await call_next(context)


__all__ = ["McpOpenTelemetryMiddleware"]
