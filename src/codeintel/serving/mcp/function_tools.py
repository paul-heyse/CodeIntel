"""Function and graph MCP tools registered from OperationSpec."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, iter_operation_specs

FUNCTION_TOOL_CATEGORIES: set[str] = {"functions", "graph", "files", "function"}


class _ModelLike(Protocol):
    def model_dump(self) -> dict[str, object]:
        ...


def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register function- and graph-related MCP tools based on OperationSpec."""

    def _register_tool_for_spec(spec: OperationSpec) -> None:
        backend_attr = getattr(backend, spec.backend_method, None)
        if not callable(backend_attr):
            message = (
                f"Backend {backend!r} does not implement method {spec.backend_method!r} "
                f"for OperationSpec id={spec.id!r}"
            )
            raise TypeError(message)
        backend_method: Callable[..., object] = backend_attr
        if spec.tool_name is None:
            message = f"OperationSpec {spec.id} is missing a tool name"
            raise ValueError(message)

        @_wrap
        def _tool(**kwargs: object) -> dict[str, object] | dict[str, ProblemDetail]:
            response = backend_method(**kwargs)
            return cast("_ModelLike", response).model_dump()

        _tool.__name__ = spec.tool_name
        _tool.__doc__ = spec.description or spec.summary

        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(_tool)

    for spec in iter_operation_specs():
        if spec.category not in FUNCTION_TOOL_CATEGORIES:
            continue
        if spec.tool_name is None:
            continue
        _register_tool_for_spec(spec)


__all__ = ["register_function_tools"]
