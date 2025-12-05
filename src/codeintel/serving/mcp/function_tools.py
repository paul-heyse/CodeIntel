"""Function and graph MCP tools registered from Operation.

Note: This module is maintained for backward compatibility. The core tool
building logic has been consolidated in ``tool_builder.py``. New code should
use ``register_tools_for_category`` from ``tools_base.py`` or the unified
``register_tools`` function.

See Also
--------
- ``codeintel.serving.mcp.tool_builder`` : Unified tool building
- ``codeintel.serving.mcp.tools_base`` : Top-level registration
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from mcp.server.fastmcp import FastMCP

from codeintel.serving.auto_pipeline import ensure_prereqs_for_mcp, is_auto_pipeline_enabled
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
from codeintel.serving.operations import Operation, iter_operations
from codeintel.serving.services.errors import generate_correlation_id

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend

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
    spec: Operation,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
) -> Callable[..., dict[str, object] | dict[str, ProblemDetail]]:
    backend_attr = getattr(backend, spec.backend_method, None)
    if not callable(backend_attr):
        message = (
            f"Backend {backend!r} does not implement method {spec.backend_method!r} "
            f"for Operation id={spec.id!r}"
        )
        raise TypeError(message)
    backend_method: Callable[..., object] = backend_attr
    model_cls = cast("ResponseFactory | None", getattr(models, spec.output_model_name, None))

    @_wrap
    def _tool(**kwargs: object) -> dict[str, object] | dict[str, ProblemDetail]:
        # Check for auto-pipeline prerequisites
        # We check gateway attribute presence as a proxy for QueryBackend
        if is_auto_pipeline_enabled() and config is not None and hasattr(backend, "gateway"):
            ensure_prereqs_for_mcp(
                op_id=spec.id,
                config=config,
                backend=cast("QueryBackend", backend),
            )

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


def register_function_tools(
    mcp: FastMCP,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
) -> None:
    """Register function- and graph-related MCP tools based on Operation.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    """
    for spec in iter_operations():
        if spec.category not in FUNCTION_TOOL_CATEGORIES or spec.tool_name is None:
            continue
        tool = _build_function_tool(spec, backend, config)
        tool.__name__ = spec.tool_name
        tool.__doc__ = spec.description or spec.summary
        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(tool)


__all__ = ["register_function_tools"]
