"""Architecture and subsystem MCP tools (injectable for tests and utilities)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from codeintel.graphs.core.protocol import GraphPluginPlan
from codeintel.graphs.core.registry import (
    DependencyPolicy,
    PlanningOptions,
    SelectionPolicy,
    plan_graph_plugins,
)
from codeintel.serving import domain_models as dm
from codeintel.serving.auto_pipeline import ensure_prereqs_for_mcp, is_auto_pipeline_enabled
from codeintel.serving.context import (
    RequestContext,
    reset_current_request_context,
    set_current_request_context,
)
from codeintel.serving.mcp import models as mcp_models
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    FunctionArchitectureResponse,
    GraphPlanResponse,
    ModuleArchitectureResponse,
    ModuleSubsystemResponse,
    ProblemDetail,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
)
from codeintel.serving.mcp.serialization import (
    ResponseFactory,
    SupportsFromDomain,
    SupportsModelDump,
    SupportsModelValidate,
)
from codeintel.serving.mcp.tool_builder import McpToolRegistrar
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.operations import Operation, get_operation
from codeintel.serving.services.errors import generate_correlation_id

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend


ModelResolver = Callable[[str], ResponseFactory | None]
PlanBuilder = Callable[
    [tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None, PlanningOptions],
    object,
]
PrereqRunner = Callable[[str, "ServingConfig", "QueryBackend"], object]


@dataclass
class ArchitectureToolOptions:
    """Optional overrides for architecture tool registration."""

    operations: Iterable[Operation] | None = None
    model_resolver: ModelResolver | None = None
    plan_builder: PlanBuilder | None = None
    prereq_runner: PrereqRunner | None = None


@dataclass(frozen=True)
class _RegistrationContext:
    backend: QueryBackendOrService
    config: ServingConfig | None
    prereq_runner: PrereqRunner


def _require_spec(op_id: str) -> Operation:
    spec = get_operation(op_id)
    if spec is None:
        message = f"Operation {op_id} is not registered"
        raise ValueError(message)
    return spec


def _load_architecture_specs(operations: Iterable[Operation] | None) -> dict[str, Operation]:
    by_id: dict[str, Operation] = {spec.id: spec for spec in operations} if operations else {}
    specs: dict[str, Operation] = {
        "subsystems.list": by_id.get("subsystems.list") or _require_spec("subsystems.list"),
        "subsystems.module_memberships": by_id.get("subsystems.module_memberships")
        or _require_spec("subsystems.module_memberships"),
        "subsystems.detail": by_id.get("subsystems.detail") or _require_spec("subsystems.detail"),
        "subsystems.search": by_id.get("subsystems.search") or _require_spec("subsystems.search"),
        "subsystems.summarize": by_id.get("subsystems.summarize")
        or _require_spec("subsystems.summarize"),
        "ide.hints": by_id.get("ide.hints") or _require_spec("ide.hints"),
        "architecture.function": by_id.get("architecture.function")
        or _require_spec("architecture.function"),
        "architecture.module": by_id.get("architecture.module")
        or _require_spec("architecture.module"),
        "graph.plugins.plan": by_id.get("graph.plugins.plan")
        or _require_spec("graph.plugins.plan"),
    }
    expected_names = {
        "subsystems.list": "list_subsystems",
        "subsystems.module_memberships": "get_module_subsystems",
        "subsystems.detail": "get_subsystem_modules",
        "subsystems.search": "search_subsystems",
        "subsystems.summarize": "summarize_subsystem",
        "ide.hints": "get_file_hints",
        "architecture.function": "get_function_architecture",
        "architecture.module": "get_module_architecture",
        "graph.plugins.plan": "graph_plugin_plan",
    }
    for op_id, tool_name in expected_names.items():
        spec = specs[op_id]
        if spec.tool_name != tool_name:
            message = f"Operation {op_id} has mismatched tool name"
            raise ValueError(message)
    return specs


def _invoke_with_request_context[T](
    backend: QueryBackendOrService,
    operation_id: str,
    func: Callable[[], T],
    *,
    dataset: str | None = None,
    graph_scope: object | None = None,
) -> T:
    correlation_id = generate_correlation_id()
    ctx = RequestContext(
        correlation_id=correlation_id,
        transport="mcp",
        operation=operation_id,
        dataset=dataset,
        repo=getattr(backend, "repo", None),
        commit=getattr(backend, "commit", None),
        snapshot=None,
        graph_scope=graph_scope,
        client_id=None,
        user_agent=None,
    )
    token = set_current_request_context(ctx)
    try:
        return func()
    finally:
        reset_current_request_context(token)


def _default_model_resolver(name: str) -> ResponseFactory | None:
    return cast(
        "ResponseFactory | None",
        getattr(mcp_models, name, None) or getattr(globals(), name, None),
    )


def _default_plan_builder(
    names: tuple[str, ...] | None,
    enabled: tuple[str, ...] | None,
    disabled: tuple[str, ...] | None,
    plan_options: PlanningOptions,
) -> object:
    return plan_graph_plugins(
        plugin_names=names,
        enabled=enabled,
        disabled=disabled,
        plan_options=plan_options,
    )


def _serialize_payload(
    payload: object,
    model_cls: ResponseFactory | None,
) -> dict[str, object]:
    if hasattr(payload, "model_dump"):
        model = cast("SupportsModelDump", payload)
        return model.model_dump()
    if model_cls is not None:
        if hasattr(model_cls, "from_domain"):
            model = cast("SupportsFromDomain", model_cls).from_domain(payload)
            return model.model_dump()
        validator = cast("SupportsModelValidate", model_cls).model_validate
        model = validator(payload)
        return model.model_dump()
    return cast("dict[str, object]", payload)


def _require_backend_method(
    spec: Operation, backend: QueryBackendOrService
) -> Callable[..., object]:
    backend_attr_obj = getattr(backend, spec.backend_method, None)
    if not callable(backend_attr_obj):
        message = (
            f"Backend {backend!r} does not implement method {spec.backend_method!r} "
            f"for Operation id={spec.id!r}"
        )
        raise TypeError(message)
    return backend_attr_obj


def _run_prereqs_if_needed(
    op_id: str,
    ctx: _RegistrationContext,
) -> None:
    if ctx.config is None:
        return
    if not is_auto_pipeline_enabled():
        return
    if not hasattr(ctx.backend, "gateway"):
        return
    ctx.prereq_runner(op_id, ctx.config, cast("QueryBackend", ctx.backend))


def _graph_plan_from_plugin_plan(plan: object) -> dm.GraphPlan:
    plan_any = cast(Any, plan)
    metadata_entries = []
    for plugin in plan_any.plugins:
        meta = plugin.metadata
        resource_hints = getattr(meta, "resource_hints", None)
        metadata_entries.append(
            (
                meta.name,
                {
                    "stage": meta.stage,
                    "severity": meta.severity,
                    "requires_isolation": getattr(meta, "isolation_kind", "none") != "none",
                    "isolation_kind": getattr(meta, "isolation_kind", "none"),
                    "scope_aware": getattr(meta, "supports_incremental", False),
                    "supported_scopes": (),
                    "description": meta.description,
                    "enabled_by_default": getattr(meta, "enabled_by_default", False),
                    "depends_on": getattr(meta, "depends_on", ()),
                    "provides": getattr(meta, "provides", ()),
                    "requires": getattr(meta, "requires", ()),
                    "resource_hints": (
                        {
                            "max_runtime_ms": resource_hints.max_runtime_ms,
                            "max_memory_mb": resource_hints.max_memory_mb,
                        }
                        if resource_hints is not None
                        else None
                    ),
                    "options_model": (
                        getattr(meta.options_model, "__name__", None)
                        if getattr(meta, "options_model", None)
                        else None
                    ),
                    "options_default": getattr(meta, "options_default", None),
                    "version_hash": getattr(meta, "version_hash", None),
                    "contract_checkers": 0,
                    "config_schema_ref": getattr(meta, "config_schema_ref", None),
                    "row_count_tables": getattr(meta, "row_count_tables", ()),
                    "cache_populates": getattr(meta, "cache_populates", ()),
                    "cache_consumes": getattr(meta, "cache_consumes", ()),
                },
            )
        )
    dep_graph = {name: tuple(deps) for name, deps in plan_any.dep_graph.items()}
    skipped = cast(
        "list[dict[str, object]]",
        [{"name": entry.name, "reason": entry.reason} for entry in plan_any.skipped_plugins],
    )
    return dm.GraphPlan(
        plan_id=str(plan_any.plan_id),
        ordered_plugins=plan_any.ordered_names,
        skipped_plugins=skipped,
        dep_graph=dep_graph,
        plugin_metadata=dict(metadata_entries),
    )


def _register_graph_plugin_plan_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    options: ArchitectureToolOptions,
) -> None:
    model_cls = (options.model_resolver or _default_model_resolver)(spec.output_model_name)
    planner = options.plan_builder or _default_plan_builder

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def graph_plugin_plan(
        names: list[str] | None = None,
        enable: list[str] | None = None,
        disable: list[str] | None = None,
        *,
        planning: dict[str, object] | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        """
        Compute graph metric plugin execution plan with ordering and dep graph.

        Parameters
        ----------
        names
            Explicit plugin names to plan (used when enable is not provided).
        enable
            Ordered list of plugins to enable (overrides defaults when provided).
        disable
            Plugins to drop from the selected set.
        planning
            Optional overrides:
                - allow_missing_dependencies (bool, default True)
                - dependency_policy ("strict" | "skip", default "skip")
                - selection_policy ("lenient" | "strict", default "lenient")

        Returns
        -------
        dict[str, object] | dict[str, ProblemDetail]
            Plan payload with ordering, skips, and dependency graph or an error detail.
        """

        def _build_response() -> dict[str, object]:
            planning_opts = planning or {}
            allow_missing = bool(planning_opts.get("allow_missing_dependencies", True))
            dependency_policy = DependencyPolicy(planning_opts.get("dependency_policy", "skip"))
            selection_policy = SelectionPolicy(planning_opts.get("selection_policy", "lenient"))
            plan_obj = planner(
                tuple(names) if names else None,
                tuple(enable) if enable else None,
                tuple(disable) if disable else None,
                PlanningOptions(
                    allow_missing_dependencies=allow_missing,
                    dependency_policy=dependency_policy,
                    selection_policy=selection_policy,
                ),
            )
            graph_plan = _graph_plan_from_plugin_plan(plan_obj)
            if model_cls is None:
                return _serialize_payload(graph_plan, GraphPlanResponse)
            return _serialize_payload(graph_plan, model_cls)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _build_response,
        )


def _register_function_architecture_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def get_function_architecture(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(goid_h128=goid_h128)
            return _serialize_payload(result, model_cls or FunctionArchitectureResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_module_architecture_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def get_module_architecture(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(module=module)
            return _serialize_payload(result, model_cls or ModuleArchitectureResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_list_subsystems_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def list_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(limit=limit, role=role, q=q)
            return _serialize_payload(result, model_cls or SubsystemSummaryResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_module_subsystems_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def get_module_subsystems(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(module=module)
            return _serialize_payload(result, model_cls or ModuleSubsystemResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_file_hints_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def get_file_hints(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(rel_path=rel_path)
            return _serialize_payload(result, model_cls or FileHintsResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_subsystem_modules_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def get_subsystem_modules(
        subsystem_id: str, module_limit: int | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(
                subsystem_id=subsystem_id,
                module_limit=module_limit,
            )
            return _serialize_payload(result, model_cls or SubsystemModulesResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_search_subsystems_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def search_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(limit=limit, role=role, q=q)
            return _serialize_payload(result, model_cls or SubsystemSearchResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def _register_summarize_subsystem_tool(
    mcp: McpToolRegistrar,
    ctx: _RegistrationContext,
    spec: Operation,
    model_resolver: ModelResolver,
) -> None:
    model_cls = model_resolver(spec.output_model_name)
    backend_method = _require_backend_method(spec, ctx.backend)

    @mcp.tool(name=spec.tool_name, description=spec.summary)
    @_wrap
    def summarize_subsystem(
        subsystem_id: str, module_limit: int | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> dict[str, object]:
            _run_prereqs_if_needed(spec.id, ctx)
            result = backend_method(
                subsystem_id=subsystem_id,
                module_limit=module_limit,
            )
            return _serialize_payload(result, model_cls or SubsystemModulesResponse)

        return _invoke_with_request_context(
            ctx.backend,
            spec.id,
            _call,
        )


def register_architecture_tools(
    mcp: McpToolRegistrar,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
    options: ArchitectureToolOptions | None = None,
) -> None:
    """Register architecture and subsystem MCP tools.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or QueryService implementation.
    config
        Optional serving config for auto-pipeline support.
    options
        Optional overrides for operations, model resolution, plan building, and prereq runner.
    """
    opts = options or ArchitectureToolOptions()
    prereq_runner = opts.prereq_runner or (
        lambda op_id, cfg, bkd: ensure_prereqs_for_mcp(op_id=op_id, config=cfg, backend=bkd)
    )
    specs = _load_architecture_specs(opts.operations)
    ctx = _RegistrationContext(
        backend=backend,
        config=config,
        prereq_runner=prereq_runner,
    )
    resolver = opts.model_resolver or _default_model_resolver
    _register_graph_plugin_plan_tool(mcp, ctx, specs["graph.plugins.plan"], opts)
    _register_function_architecture_tool(
        mcp,
        ctx,
        specs["architecture.function"],
        resolver,
    )
    _register_module_architecture_tool(
        mcp,
        ctx,
        specs["architecture.module"],
        resolver,
    )
    _register_list_subsystems_tool(
        mcp,
        ctx,
        specs["subsystems.list"],
        resolver,
    )
    _register_module_subsystems_tool(
        mcp,
        ctx,
        specs["subsystems.module_memberships"],
        resolver,
    )
    _register_file_hints_tool(
        mcp,
        ctx,
        specs["ide.hints"],
        resolver,
    )
    _register_subsystem_modules_tool(
        mcp,
        ctx,
        specs["subsystems.detail"],
        resolver,
    )
    _register_search_subsystems_tool(
        mcp,
        ctx,
        specs["subsystems.search"],
        resolver,
    )
    _register_summarize_subsystem_tool(
        mcp,
        ctx,
        specs["subsystems.summarize"],
        resolver,
    )


__all__ = ["ArchitectureToolOptions", "register_architecture_tools"]
