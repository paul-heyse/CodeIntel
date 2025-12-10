"""Declarative MCP tool registration from the Operation catalog.

Tool Registration Architecture
------------------------------
Tools are registered declaratively based on the Operation catalog
(``operations/catalog.py``). Each Operation defines:

- ``id``: Unique operation identifier
- ``tool_name``: MCP tool name (if exposed as a tool)
- ``backend_method``: Method name on QueryBackend or QueryService
- ``output_model_name``: Response model class name
- ``category``: Tool category for filtering

Registration Flow
~~~~~~~~~~~~~~~~~
::

    Operation Catalog
         │
         ▼
    build_tool_from_operation()  ──▶ MCP tool function
         │
         ▼
    register_tools_for_category()  ──▶ mcp.tool()

Auto-Pipeline Support
~~~~~~~~~~~~~~~~~~~~~
When ``auto_pipeline`` is enabled, the tool builder automatically checks
for and runs missing pipeline prerequisites before executing operations.
This is controlled via the ``config`` parameter and the
``is_auto_pipeline_enabled()`` flag.

See Also
--------
- ``codeintel.serving.operations.catalog`` : Operation definitions
- ``codeintel.serving.mcp.tools_base`` : Top-level registration entry point
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

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


class _ModelLike(Protocol):
    """Protocol for objects with model_dump method."""

    def model_dump(self) -> dict[str, object]:
        """Serialize model to dictionary."""
        ...


class McpToolRegistrar(Protocol):
    """Minimal MCP registrar interface consumed by tool registration."""

    def tool(
        self,
        name: str | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorate and register an MCP tool callable."""
        ...


def _serialize_payload(
    payload: object,
    model_cls: ResponseFactory | None,
) -> dict[str, object]:
    """Serialize a response payload to a dictionary.

    Parameters
    ----------
    payload
        The response object from a backend method.
    model_cls
        Optional response model class for conversion.

    Returns
    -------
    dict[str, object]
        Serialized payload as a dictionary.
    """
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


def build_tool_from_operation(
    spec: Operation,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
    *,
    model_resolver: Callable[[str], ResponseFactory | None] | None = None,
    prereq_runner: Callable[[str, ServingConfig, QueryBackend], object] | None = None,
) -> Callable[..., dict[str, object] | dict[str, ProblemDetail]]:
    """Build an MCP tool function from an Operation specification.

    This is the core factory function that transforms an Operation definition
    into a callable MCP tool. The resulting tool handles:

    - Request context setup with correlation IDs
    - Auto-pipeline prerequisite checking
    - Response serialization via the appropriate model class
    - Error wrapping via the ``_wrap`` decorator

    Parameters
    ----------
    spec
        Operation specification defining the tool.
    backend
        Backend or service providing the implementation method.
    config
        Optional serving config for auto-pipeline support.
    model_resolver
        Optional resolver for response models (defaults to module lookup).
    prereq_runner
        Optional runner for auto-pipeline prerequisites (defaults to
        ``ensure_prereqs_for_mcp``).

    Returns
    -------
    Callable
        Tool function suitable for MCP registration.

    Raises
    ------
    TypeError
        If the backend does not implement the required method.
    """
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
    def _tool(**kwargs: object) -> dict[str, object] | dict[str, ProblemDetail]:
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


@dataclass
class ToolRegistrationOptions:
    """Optional overrides for tool registration."""

    operations: Iterable[Operation] | None = None
    model_resolver: Callable[[str], ResponseFactory | None] | None = None
    prereq_runner: Callable[[str, ServingConfig, QueryBackend], object] | None = None


def register_tools_for_category(
    mcp: McpToolRegistrar,
    backend: QueryBackendOrService,
    categories: set[str],
    config: ServingConfig | None = None,
    options: ToolRegistrationOptions | None = None,
) -> None:
    """Register MCP tools for specific categories from the Operation catalog.

    This function iterates through all operations in the catalog and registers
    those matching the specified categories as MCP tools.

    Parameters
    ----------
    mcp
        MCP registrar used to register tools.
    backend
        Backend or service providing implementations.
    categories
        Set of category names to register (e.g., {"functions", "graph"}).
    config
        Optional serving config for auto-pipeline support.
    options
        Optional overrides for operations, model resolution, and prereq runner.
    """
    opts = options or ToolRegistrationOptions()
    for spec in opts.operations or iter_operations():
        if spec.category not in categories or spec.tool_name is None:
            continue
        tool = build_tool_from_operation(
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


def register_all_tools(
    mcp: McpToolRegistrar,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
    options: ToolRegistrationOptions | None = None,
) -> None:
    """Register all MCP tools from the Operation catalog.

    This is a convenience function that registers all operations that have
    a ``tool_name`` defined. Use ``register_tools_for_category`` if you need
    to filter by specific categories.

    Parameters
    ----------
    mcp
        MCP registrar used to register tools.
    backend
        Backend or service providing implementations.
    config
        Optional serving config for auto-pipeline support.
    options
        Optional overrides for operations, model resolution, and prereq runner.
    """
    opts = options or ToolRegistrationOptions()
    for spec in opts.operations or iter_operations():
        if spec.tool_name is None:
            continue
        tool = build_tool_from_operation(
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


__all__ = [
    "McpToolRegistrar",
    "ToolRegistrationOptions",
    "build_tool_from_operation",
    "register_all_tools",
    "register_tools_for_category",
]
