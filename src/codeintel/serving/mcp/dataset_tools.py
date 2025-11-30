"""Dataset MCP tools registered from OperationSpec."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, iter_operation_specs


def register_dataset_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register dataset browsing MCP tools based on OperationSpec."""

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
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if isinstance(response, list):
                return [
                    item.model_dump() if hasattr(item, "model_dump") else item  # type: ignore[arg-type]
                    for item in response
                ]
            return response  # type: ignore[return-value]

        if spec.tool_name is None:
            message = f"OperationSpec {spec.id} is missing a tool name"
            raise ValueError(message)
        _tool.__name__ = spec.tool_name
        _tool.__doc__ = spec.description or spec.summary

        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(_tool)

    for spec in iter_operation_specs():
        if spec.category != "datasets":
            continue
        if spec.tool_name is None:
            continue
        _register_tool_for_spec(spec)


__all__ = ["register_dataset_tools"]
