"""MCP tool registration orchestrator.

Tool Registration Architecture
------------------------------
All tools are registered declaratively from the Operation catalog via
the ``tool_builder`` module. This module provides the top-level entry point
``register_tools()`` which delegates to:

1. ``register_tools_for_category()`` for standard operation-based tools
2. ``register_architecture_tools()`` for subsystem/architecture tools
3. ``register_meta_tools()`` for introspection tools

See Also
--------
- ``codeintel.serving.mcp.tool_builder`` : Core tool building logic
- ``codeintel.serving.operations.catalog`` : Operation definitions
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

from codeintel.serving.mcp.architecture_tools import register_architecture_tools
from codeintel.serving.mcp.meta_tools import register_meta_tools
from codeintel.serving.mcp.tool_builder import (
    build_tool_from_operation,
    register_all_tools,
    register_tools_for_category,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.operations.catalog import iter_registry_operations

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.tool_builder import (
        McpToolRegistrar,
    )


_STANDARD_CATEGORIES: set[str] = {"functions", "graph", "files", "profiles", "datasets"}


class _ToolMethod(Protocol):
    def __call__(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...


class _RegistrarWrapper:
    """Adapt objects exposing a compatible tool() method to McpToolRegistrar."""

    def __init__(self, tool_method: _ToolMethod) -> None:
        self._tool_method = tool_method

    def tool(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        return self._tool_method(name=name, **options)


def as_registrar(mcp: McpToolRegistrar | object) -> McpToolRegistrar:
    """Coerce an object with a tool() method into a McpToolRegistrar.

    Returns
    -------
    McpToolRegistrar
        Adapter that forwards tool registrations.

    Raises
    ------
    TypeError
        If the provided object lacks a callable tool method.
    """
    tool_method = getattr(mcp, "tool", None)
    if callable(tool_method):
        return _RegistrarWrapper(cast("_ToolMethod", tool_method))
    message = f"Provided MCP object {mcp!r} does not expose a tool registration method"
    raise TypeError(message)


def _expose_tools(mcp: object) -> None:
    """Populate mcp.tools from list_tools() (supports sync or async), with fallback."""
    tools = None
    mcp_any = cast("Any", mcp)
    list_tools_fn = getattr(mcp_any, "list_tools", None)
    if callable(list_tools_fn):
        try:
            result = list_tools_fn()
            if inspect.iscoroutine(result):
                try:
                    tools = asyncio.run(result)
                except RuntimeError:
                    tools = None
            elif inspect.isawaitable(result):
                tools = None
            else:
                tools = result
        except (AttributeError, TypeError):
            tools = None
    if tools is not None and not inspect.iscoroutine(tools):
        mcp_any.tools = tools
        return
    fallback_tools = [
        SimpleNamespace(name=op.tool_name) for op in iter_registry_operations() if op.tool_name
    ]
    mcp_any.tools = fallback_tools


def expose_tools(mcp: object) -> None:
    """Public wrapper to expose tools on an MCP object."""
    _expose_tools(mcp)


def register_tools(
    mcp: McpToolRegistrar | object,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
) -> None:
    """Register all MCP tools on the given FastMCP instance.

    This is the main entry point for MCP tool registration. It uses:

    1. ``register_tools_for_category()`` for standard operation-based tools
    2. ``register_architecture_tools()`` for subsystem/architecture tools
    3. ``register_meta_tools()`` for introspection tools

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or any QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    """
    registrar = as_registrar(mcp)

    register_tools_for_category(registrar, backend, _STANDARD_CATEGORIES, config)

    register_architecture_tools(registrar, backend, config)

    register_meta_tools(registrar, backend)

    _expose_tools(mcp)


__all__ = [
    "QueryBackendOrService",
    "as_registrar",
    "build_tool_from_operation",
    "expose_tools",
    "register_all_tools",
    "register_tools",
    "register_tools_for_category",
]
