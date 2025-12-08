"""Dataset MCP tools registered from Operation.

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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
    SupportsModelDump,
    SupportsModelValidate,
)
from codeintel.serving.mcp.tool_builder import McpToolRegistrar
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.operations import Operation, iter_operations
from codeintel.serving.services.errors import generate_correlation_id

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend


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
    spec: Operation,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
    *,
    model_resolver: Callable[[str], ResponseFactory | None] | None = None,
    prereq_runner: Callable[[str, ServingConfig, QueryBackend], object] | None = None,
) -> Callable[..., list[dict[str, object]] | dict[str, object] | dict[str, ProblemDetail]]:
    backend_attr = getattr(backend, spec.backend_method, None)
    if not callable(backend_attr):
        message = (
            f"Backend {backend!r} does not implement method {spec.backend_method!r} "
            f"for Operation id={spec.id!r}"
        )
        raise TypeError(message)
    backend_method: Callable[..., object] = backend_attr
    resolver = model_resolver or (lambda name: getattr(models, name, None))
    model_cls = cast("ResponseFactory | None", resolver(spec.output_model_name))
    run_prereqs = prereq_runner or (
        lambda op_id, cfg, bkd: ensure_prereqs_for_mcp(op_id=op_id, config=cfg, backend=bkd)
    )

    @_wrap
    def _tool(
        **kwargs: object,
    ) -> list[dict[str, object]] | dict[str, object] | dict[str, ProblemDetail]:
        # Check for auto-pipeline prerequisites
        if is_auto_pipeline_enabled() and config is not None and hasattr(backend, "gateway"):
            run_prereqs(spec.id, config, cast("QueryBackend", backend))

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


@dataclass
class DatasetToolOptions:
    """Optional overrides for dataset tool registration."""

    operations: Iterable[Operation] | None = None
    model_resolver: Callable[[str], ResponseFactory | None] | None = None
    prereq_runner: Callable[[str, ServingConfig, QueryBackend], object] | None = None


def register_dataset_tools(
    mcp: McpToolRegistrar,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
    options: DatasetToolOptions | None = None,
) -> None:
    """Register dataset browsing MCP tools based on Operation.

    Parameters
    ----------
    mcp
        FastMCP-compatible registrar to register tools against.
    backend
        Concrete MCP backend or QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    options
        Optional overrides for operations, model resolution, and prereq runner.
    """
    opts = options or DatasetToolOptions()
    for spec in opts.operations or iter_operations():
        if spec.category != "datasets" or spec.tool_name is None:
            continue
        tool = _build_dataset_tool(
            spec,
            backend,
            config,
            model_resolver=opts.model_resolver,
            prereq_runner=opts.prereq_runner,
        )
        tool.__name__ = spec.tool_name
        tool.__doc__ = spec.description or spec.summary
        mcp.tool(
            name=spec.tool_name,
            description=spec.summary,
        )(tool)


__all__ = ["DatasetToolOptions", "register_dataset_tools"]
