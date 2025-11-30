"""Common MCP tool registration helpers and error wrapping."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService


def register_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """
    Register all MCP tools on the given FastMCP instance.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or any QueryService implementation.
    """
    register_function_tools(mcp, backend)
    register_profile_tools(mcp, backend)
    register_architecture_tools(mcp, backend)
    register_dataset_tools(mcp, backend)


__all__ = ["QueryBackendOrService", "register_tools"]
