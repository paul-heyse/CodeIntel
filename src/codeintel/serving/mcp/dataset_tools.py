"""Dataset MCP tools registered from OperationSpec."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

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
    SupportsModelDump,
    SupportsModelValidate,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.registry import OperationSpec, iter_operation_specs
from codeintel.serving.services.errors import generate_correlation_id


def _serialize_payload(
    payload: object,
    model_cls: ResponseFactory | None,
) -> dict[str, object]:
    if hasattr(payload, "model_dump"):
        return cast("SupportsModelDump", payload).model_dump()
    if model_cls is not None:
        if hasattr(model_cls, "from_domain"):
            from_domain = cast("SupportsFromDomain", model_cls).from_domain
            return from_domain(payload).model_dump()
        validator = cast("SupportsModelValidate", model_cls).model_validate
        return validator(payload).model_dump()
    return cast("dict[str, object]", payload)


def _serialize_list_payload(
    payload: list[object], model_cls: ResponseFactory | None
) -> list[dict[str, object] | object]:
    if model_cls is None and all(hasattr(item, "model_dump") for item in payload):
        return [cast("SupportsModelDump", item).model_dump() for item in payload]
    return [_serialize_payload(item, model_cls) for item in payload]


def _build_dataset_tool(
    spec: OperationSpec,
    backend: QueryBackendOrService,
) -> Callable[..., list[dict[str, object]] | dict[str, object] | dict[str, ProblemDetail]]:
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
    def _tool(
        **kwargs: object,
    ) -> list[dict[str, object]] | dict[str, object] | dict[str, ProblemDetail]:
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
            graph_scope=None,
            client_id=None,
            user_agent=None,
        )
        token = set_current_request_context(ctx)
        try:
            response = backend_method(**kwargs)
            if isinstance(response, list):
                return cast("list[dict[str, object]]", _serialize_list_payload(response, model_cls))
            return _serialize_payload(response, model_cls)
        finally:
            reset_current_request_context(token)

    return cast(
        "Callable[..., list[dict[str, object]] | dict[str, object] | dict[str, ProblemDetail]]",
        _tool,
    )


def register_dataset_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register dataset browsing MCP tools based on OperationSpec."""
    for spec in iter_operation_specs():
        if spec.category != "datasets" or spec.tool_name is None:
            continue
        tool = _build_dataset_tool(spec, backend)
        tool.__name__ = spec.tool_name
        tool.__doc__ = spec.description or spec.summary
        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(tool)


__all__ = ["register_dataset_tools"]
