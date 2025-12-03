"""MCP tool registry shim."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.tools_base import register_tools as _register_tools
from codeintel.serving.services.query_service import QueryService

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig


def register_tools(
    mcp: FastMCP,
    backend: QueryBackend | QueryService,
    config: ServingConfig | None = None,
) -> None:
    """Register all MCP tools on the given FastMCP instance.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    """
    _register_tools(mcp, backend, config)


__all__ = ["register_tools"]
