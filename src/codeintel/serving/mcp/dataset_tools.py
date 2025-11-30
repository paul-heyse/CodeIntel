"""Dataset MCP tools registered from OperationSpec."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, iter_operation_specs


class _ModelLike(Protocol):
    def model_dump(self) -> dict[str, object]:
        ...


def register_dataset_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register dataset browsing MCP tools based on OperationSpec."""

    def _register_tool_for_spec(spec: OperationSpec) -> None:
        backend_attr = getattr(backend, spec.backend_method, None)
        if not callable(backend_attr):
            message = (
                f"Backend {backend!r} does not implement method {spec.backend_method!r} "
                f"for OperationSpec id={spec.id!r}"
            )
            raise TypeError(message)
        backend_method: Callable[..., object] = backend_attr

        @_wrap
        def _tool(
            **kwargs: object,
        ) -> list[dict[str, object]] | dict[str, object] | dict[str, ProblemDetail]:
            response = backend_method(**kwargs)
            if isinstance(response, list):
                return [
                    cast("_ModelLike", item).model_dump() if hasattr(item, "model_dump") else item
                    for item in response
                ]
            if hasattr(response, "model_dump"):
                return cast("_ModelLike", response).model_dump()
            return cast("dict[str, object]", response)

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
