"""MCP tool registration orchestrator.

Tool Registration Architecture
------------------------------
All tools are now registered declaratively from the Operation catalog via
the ``tool_builder`` module. This module provides the top-level entry point
that delegates to category-specific registration functions.

The per-category registration functions still exist for backward compatibility
and granular control, but the unified ``register_tools`` function is the
preferred entry point for new code.

See Also
--------
- ``codeintel.serving.mcp.tool_builder`` : Core tool building logic
- ``codeintel.serving.operations.catalog`` : Operation definitions
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.dataset_tools import register_dataset_tools
from codeintel.serving.mcp.function_tools import register_function_tools
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.serving.mcp.profile_tools import register_profile_tools
from codeintel.serving.mcp.tool_builder import (
    build_tool_from_operation,
    register_all_tools,
    register_tools_for_category,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import iter_registry_operations

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig


def register_tools(
    mcp: FastMCP,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
) -> None:
    """Register all MCP tools on the given FastMCP instance.

    This is the main entry point for MCP tool registration. It delegates
    to category-specific registration functions which use the unified
    ``tool_builder`` module.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or any QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    """
    register_function_tools(mcp, backend, config)
    register_profile_tools(mcp, backend, config)
    register_architecture_tools(mcp, backend, config)
    register_dataset_tools(mcp, backend, config)
    register_meta_tools(mcp, backend)
    # Expose registered tools for callers that inspect `mcp.tools` in tests/utilities.
    tools = None
    if not inspect.iscoroutinefunction(getattr(mcp, "list_tools", None)):
        try:
            tools = mcp.list_tools()
        except (AttributeError, TypeError):
            tools = None
    if tools is not None and not inspect.iscoroutine(tools):
        cast("Any", mcp).tools = tools
        return
    fallback_tools = [
        SimpleNamespace(name=op.tool_name) for op in iter_registry_operations() if op.tool_name
    ]
    cast("Any", mcp).tools = fallback_tools


__all__ = [
    "QueryBackendOrService",
    "build_tool_from_operation",
    "register_all_tools",
    "register_tools",
    "register_tools_for_category",
]
