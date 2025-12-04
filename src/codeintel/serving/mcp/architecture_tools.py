"""Architecture and subsystem MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from codeintel.graphs.core.registry import plan_graph_plugins
from codeintel.serving.context import (
    RequestContext,
    reset_current_request_context,
    set_current_request_context,
)
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    FunctionArchitectureResponse,
    GraphPlanPluginMetadata,
    GraphPlanResponse,
    GraphPlanSkipped,
    ModuleArchitectureResponse,
    ModuleSubsystemResponse,
    ProblemDetail,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
)
from codeintel.serving.mcp.tool_utils import QueryBackendOrService, _wrap
from codeintel.serving.operations import Operation, get_operation
from codeintel.serving.services.errors import generate_correlation_id

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig


def _require_spec(op_id: str) -> Operation:
    spec = get_operation(op_id)
    if spec is None:
        message = f"Operation {op_id} is not registered"
        raise ValueError(message)
    return spec


def _load_architecture_specs() -> dict[str, Operation]:
    specs: dict[str, Operation] = {
        "subsystems.list": _require_spec("subsystems.list"),
        "subsystems.module_memberships": _require_spec("subsystems.module_memberships"),
        "subsystems.detail": _require_spec("subsystems.detail"),
        "subsystems.search": _require_spec("subsystems.search"),
        "subsystems.summarize": _require_spec("subsystems.summarize"),
        "ide.hints": _require_spec("ide.hints"),
        "architecture.function": _require_spec("architecture.function"),
        "architecture.module": _require_spec("architecture.module"),
        "graph.plugins.plan": _require_spec("graph.plugins.plan"),
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


def _register_graph_plugin_plan_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def graph_plugin_plan(
        names: list[str] | None = None,
        enable: list[str] | None = None,
        disable: list[str] | None = None,
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

        Returns
        -------
        dict[str, object] | dict[str, ProblemDetail]
            Plan payload with ordering, skips, and dependency graph or an error detail.
        """

        def _build_response() -> GraphPlanResponse:
            plan = plan_graph_plugins(
                plugin_names=tuple(names) if names else None,
                enabled=tuple(enable) if enable else None,
                disabled=tuple(disable) if disable else None,
            )
            metadata = {
                plugin.metadata.name: GraphPlanPluginMetadata(
                    stage=plugin.metadata.stage,
                    severity=plugin.metadata.severity,
                    requires_isolation=plugin.metadata.isolation_kind != "none",
                    isolation_kind=plugin.metadata.isolation_kind,
                    scope_aware=plugin.metadata.supports_incremental,
                    supported_scopes=(),
                    description=plugin.metadata.description,
                    enabled_by_default=plugin.metadata.enabled_by_default,
                    depends_on=plugin.metadata.depends_on,
                    provides=plugin.metadata.provides,
                    requires=plugin.metadata.requires,
                    resource_hints=(
                        {
                            "max_runtime_ms": plugin.metadata.resource_hints.max_runtime_ms,
                            "max_memory_mb": plugin.metadata.resource_hints.max_memory_mb,
                        }
                        if plugin.metadata.resource_hints is not None
                        else None
                    ),
                    options_model=(
                        plugin.metadata.options_model.__name__
                        if plugin.metadata.options_model
                        else None
                    ),
                    options_default=plugin.metadata.options_default,
                    version_hash=plugin.metadata.version_hash,
                    contract_checkers=0,
                    config_schema_ref=plugin.metadata.config_schema_ref,
                    row_count_tables=plugin.metadata.row_count_tables,
                    cache_populates=plugin.metadata.cache_populates,
                    cache_consumes=plugin.metadata.cache_consumes,
                )
                for plugin in plan.plugins
            }
            return GraphPlanResponse(
                plan_id=plan.plan_id,
                ordered_plugins=plan.ordered_names,
                skipped_plugins=tuple(
                    GraphPlanSkipped(name=skipped.name, reason=skipped.reason)
                    for skipped in plan.skipped_plugins
                ),
                dep_graph={name: tuple(deps) for name, deps in plan.dep_graph.items()},
                plugin_metadata=metadata,
            )

        response = _invoke_with_request_context(
            backend,
            "graph.plugins.plan",
            _build_response,
        )
        return response.model_dump()


def _register_function_architecture_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def get_function_architecture(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> FunctionArchitectureResponse:
            result = backend.get_function_architecture(goid_h128=goid_h128)
            if isinstance(result, FunctionArchitectureResponse):
                return result
            return FunctionArchitectureResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "architecture.function",
            _call,
        )
        return response.model_dump()


def _register_module_architecture_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def get_module_architecture(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> ModuleArchitectureResponse:
            result = backend.get_module_architecture(module=module)
            if isinstance(result, ModuleArchitectureResponse):
                return result
            return ModuleArchitectureResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "architecture.module",
            _call,
        )
        return response.model_dump()


def _register_list_subsystems_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def list_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> SubsystemSummaryResponse:
            result = backend.list_subsystems(limit=limit, role=role, q=q)
            if isinstance(result, SubsystemSummaryResponse):
                return result
            return SubsystemSummaryResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "subsystems.list",
            _call,
        )
        return response.model_dump()


def _register_module_subsystems_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def get_module_subsystems(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> ModuleSubsystemResponse:
            result = backend.get_module_subsystems(module=module)
            if isinstance(result, ModuleSubsystemResponse):
                return result
            return ModuleSubsystemResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "subsystems.module_memberships",
            _call,
        )
        return response.model_dump()


def _register_file_hints_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def get_file_hints(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> FileHintsResponse:
            result = backend.get_file_hints(rel_path=rel_path)
            if isinstance(result, FileHintsResponse):
                return result
            return FileHintsResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "ide.hints",
            _call,
        )
        return response.model_dump()


def _register_subsystem_modules_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def get_subsystem_modules(
        subsystem_id: str, module_limit: int | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> SubsystemModulesResponse:
            result = backend.get_subsystem_modules(
                subsystem_id=subsystem_id,
                module_limit=module_limit,
            )
            if isinstance(result, SubsystemModulesResponse):
                return result
            return SubsystemModulesResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "subsystems.detail",
            _call,
        )
        return response.model_dump()


def _register_search_subsystems_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def search_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> SubsystemSearchResponse:
            result = backend.search_subsystems(limit=limit, role=role, q=q)
            if isinstance(result, SubsystemSearchResponse):
                return result
            return SubsystemSearchResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "subsystems.search",
            _call,
        )
        return response.model_dump()


def _register_summarize_subsystem_tool(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    @mcp.tool()
    @_wrap
    def summarize_subsystem(
        subsystem_id: str, module_limit: int | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        def _call() -> SubsystemModulesResponse:
            result = backend.summarize_subsystem(
                subsystem_id=subsystem_id,
                module_limit=module_limit,
            )
            if isinstance(result, SubsystemModulesResponse):
                return result
            return SubsystemModulesResponse.from_domain(result)

        response = _invoke_with_request_context(
            backend,
            "subsystems.summarize",
            _call,
        )
        return response.model_dump()


def register_architecture_tools(
    mcp: FastMCP,
    backend: QueryBackendOrService,
    config: ServingConfig | None = None,
) -> None:
    """Register architecture and subsystem MCP tools.

    Parameters
    ----------
    mcp
        FastMCP instance to register tools against.
    backend
        Concrete MCP backend or QueryService implementation.
    config
        Optional serving config for auto-pipeline support (reserved for future).
    """
    _load_architecture_specs()
    # Note: config is reserved for future auto-pipeline integration
    _ = config  # Unused for now
    _register_graph_plugin_plan_tool(mcp, backend)
    _register_function_architecture_tool(mcp, backend)
    _register_module_architecture_tool(mcp, backend)
    _register_list_subsystems_tool(mcp, backend)
    _register_module_subsystems_tool(mcp, backend)
    _register_file_hints_tool(mcp, backend)
    _register_subsystem_modules_tool(mcp, backend)
    _register_search_subsystems_tool(mcp, backend)
    _register_summarize_subsystem_tool(mcp, backend)


__all__ = ["register_architecture_tools"]
