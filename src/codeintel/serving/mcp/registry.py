"""MCP tool registry shim."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.tools_base import register_tools as _register_tools
from codeintel.serving.services.query_service import QueryService


def register_tools(mcp: FastMCP, backend: QueryBackend | QueryService) -> None:
    """Register all MCP tools on the given FastMCP instance."""
    _register_tools(mcp, backend)


__all__ = ["register_tools"]
