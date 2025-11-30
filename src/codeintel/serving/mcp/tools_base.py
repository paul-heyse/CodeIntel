"""Common MCP tool registration helpers and error wrapping."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.registry import iter_operation_specs


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
    register_meta_tools(mcp, backend)
    # Expose registered tools for callers that inspect `mcp.tools` in tests/utilities.
    tools = None
    if not inspect.iscoroutinefunction(getattr(mcp, "list_tools", None)):
        try:
            tools = mcp.list_tools()
        except (AttributeError, TypeError):
            tools = None
    if tools is not None and not inspect.iscoroutine(tools):
        mcp.tools = tools  # type: ignore[attr-defined]
        return
    mcp.tools = [
        SimpleNamespace(name=spec.tool_name) for spec in iter_operation_specs() if spec.tool_name
    ]


__all__ = ["QueryBackendOrService", "register_tools"]
