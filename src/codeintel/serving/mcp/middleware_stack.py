"""FastMCP middleware assembly for CodeIntel serving.

This module defines the canonical middleware stack for the MCP surface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastmcp.server.middleware.caching import (
    CallToolSettings,
    GetPromptSettings,
    ListPromptsSettings,
    ListResourcesSettings,
    ListToolsSettings,
    ReadResourceSettings,
    ResponseCachingMiddleware,
)
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import DetailedTimingMiddleware

from codeintel.serving.mcp.middleware_errors import CodeIntelErrorMappingMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext

    from codeintel.serving.settings import ServingSettings

LOG = logging.getLogger(__name__)


def _get_rate_limit_client_id(context: MiddlewareContext[object]) -> str:
    fastmcp_context = context.fastmcp_context
    if fastmcp_context is None:
        return "global"
    try:
        session_id_obj = getattr(fastmcp_context, "session_id", None)
    except RuntimeError:
        return "bootstrap"
    if isinstance(session_id_obj, str) and session_id_obj:
        return session_id_obj
    return "global"


def build_mcp_middleware(settings: ServingSettings) -> Sequence[Middleware]:
    """Build the FastMCP middleware stack.

    Parameters
    ----------
    settings
        Serving settings controlling MCP middleware configuration.

    Returns
    -------
    Sequence[Middleware]
        Middleware stack to pass to `FastMCP(..., middleware=...)`.
    """
    middleware: list[Middleware] = []

    middleware.append(CodeIntelErrorMappingMiddleware())

    if settings.mcp_enable_structured_logging:
        middleware.append(StructuredLoggingMiddleware())

    middleware.append(DetailedTimingMiddleware())

    middleware.append(
        RateLimitingMiddleware(
            max_requests_per_second=settings.mcp_rate_limit_rps,
            burst_capacity=settings.mcp_rate_limit_burst,
            get_client_id=_get_rate_limit_client_id,
            global_limit=False,
        )
    )

    if settings.mcp_cache_listings:
        ttl = max(settings.mcp_cache_listings_ttl_seconds, 1)
        middleware.append(
            ResponseCachingMiddleware(
                list_tools_settings=ListToolsSettings(enabled=True, ttl=ttl),
                list_resources_settings=ListResourcesSettings(enabled=True, ttl=ttl),
                list_prompts_settings=ListPromptsSettings(enabled=True, ttl=ttl),
                read_resource_settings=ReadResourceSettings(enabled=False),
                get_prompt_settings=GetPromptSettings(enabled=False),
                call_tool_settings=CallToolSettings(enabled=False),
            )
        )

    return middleware


__all__ = ["build_mcp_middleware"]
