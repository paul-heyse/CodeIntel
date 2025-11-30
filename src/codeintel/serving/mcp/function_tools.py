"""Function and graph MCP tools registered from OperationSpec."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, iter_operation_specs

FUNCTION_TOOL_CATEGORIES: set[str] = {"functions", "graph", "files"}


def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register function- and graph-related MCP tools based on OperationSpec."""

    def _register_tool_for_spec(spec: OperationSpec) -> None:
        backend_method = getattr(backend, spec.backend_method, None)
        if backend_method is None:
            message = (
                f"Backend {backend!r} does not implement method {spec.backend_method!r} "
                f"for OperationSpec id={spec.id!r}"
            )
            raise RuntimeError(message)

        @_wrap
        def _tool(**kwargs: object) -> dict[str, object] | dict[str, ProblemDetail]:
            response = backend_method(**kwargs)
            return response.model_dump()

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
