"""Function and graph MCP tools registered from OperationSpec."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.context import (
    RequestContext,
    reset_current_request_context,
    set_current_request_context,
)
from codeintel.serving.mcp import models
from codeintel.serving.mcp.models import ProblemDetail
from codeintel.serving.mcp.serialization import (
    ResponseFactory,
    SupportsFromDomain,
    SupportsModelValidate,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, iter_operation_specs
from codeintel.serving.services.errors import generate_correlation_id

FUNCTION_TOOL_CATEGORIES: set[str] = {"functions", "graph", "files", "function"}


class _ModelLike(Protocol):
    def model_dump(self) -> dict[str, object]: ...


def _serialize_payload(
    payload: object,
    model_cls: ResponseFactory | None,
) -> dict[str, object]:
    if hasattr(payload, "model_dump"):
        return cast("_ModelLike", payload).model_dump()
    if model_cls is not None:
        if hasattr(model_cls, "from_domain"):
            return cast(
                "_ModelLike",
                cast("SupportsFromDomain", model_cls).from_domain(payload),
            ).model_dump()
        validator = cast("SupportsModelValidate", model_cls).model_validate
        return cast("_ModelLike", validator(payload)).model_dump()
    return cast("dict[str, object]", payload)


def _build_function_tool(
    spec: OperationSpec,
    backend: QueryBackendOrService,
) -> Callable[..., dict[str, object] | dict[str, ProblemDetail]]:
    backend_attr = getattr(backend, spec.backend_method, None)
    if not callable(backend_attr):
        message = (
            f"Backend {backend!r} does not implement method {spec.backend_method!r} "
            f"for OperationSpec id={spec.id!r}"
        )
        raise TypeError(message)
    backend_method: Callable[..., object] = backend_attr
    model_cls = cast("ResponseFactory | None", getattr(models, spec.output_model_name, None))

    @_wrap
    def _tool(**kwargs: object) -> dict[str, object] | dict[str, ProblemDetail]:
        correlation_id = generate_correlation_id()
        dataset = kwargs.get("dataset_name") or kwargs.get("dataset")
        ctx = RequestContext(
            correlation_id=correlation_id,
            transport="mcp",
            operation=spec.id,
            dataset=str(dataset) if dataset is not None else None,
            repo=getattr(backend, "repo", None),
            commit=getattr(backend, "commit", None),
            snapshot=None,
            graph_scope=kwargs.get("scope"),
            client_id=None,
            user_agent=None,
        )
        token = set_current_request_context(ctx)
        try:
            response = backend_method(**kwargs)
            return _serialize_payload(response, model_cls)
        finally:
            reset_current_request_context(token)

    return cast("Callable[..., dict[str, object] | dict[str, ProblemDetail]]", _tool)


def register_function_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register function- and graph-related MCP tools based on OperationSpec."""
    for spec in iter_operation_specs():
        if spec.category not in FUNCTION_TOOL_CATEGORIES or spec.tool_name is None:
            continue
        tool = _build_function_tool(spec, backend)
        tool.__name__ = spec.tool_name
        tool.__doc__ = spec.description or spec.summary
        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(tool)


__all__ = ["register_function_tools"]
