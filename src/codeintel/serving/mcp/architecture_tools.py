"""Architecture and subsystem MCP tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from codeintel.analytics.graphs.plugins import plan_graph_metric_plugins
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
from codeintel.serving.registry import OperationSpec, get_operation_spec


def _require_spec(op_id: str) -> OperationSpec:
    spec = get_operation_spec(op_id)
    if spec is None:
        message = f"OperationSpec {op_id} is not registered"
        raise ValueError(message)
    return spec


def _load_architecture_specs() -> dict[str, OperationSpec]:
    specs: dict[str, OperationSpec] = {
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
            message = f"OperationSpec {op_id} has mismatched tool name"
            raise ValueError(message)
    return specs


def register_architecture_tools(mcp: FastMCP, backend: QueryBackendOrService) -> None:
    """Register architecture and subsystem MCP tools."""
    _load_architecture_specs()

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
        plan = plan_graph_metric_plugins(
            plugin_names=tuple(names) if names else None,
            enabled=tuple(enable) if enable else None,
            disabled=tuple(disable) if disable else None,
        )
        metadata = {
            plugin.name: GraphPlanPluginMetadata(
                stage=plugin.stage,
                severity=plugin.severity,
                requires_isolation=plugin.requires_isolation,
                isolation_kind=plugin.isolation_kind,
                scope_aware=plugin.scope_aware,
                supported_scopes=plugin.supported_scopes,
                description=plugin.description,
                enabled_by_default=plugin.enabled_by_default,
                depends_on=plugin.depends_on,
                provides=plugin.provides,
                requires=plugin.requires,
                resource_hints=(
                    {
                        "max_runtime_ms": plugin.resource_hints.max_runtime_ms,
                        "memory_mb_hint": plugin.resource_hints.memory_mb_hint,
                    }
                    if plugin.resource_hints is not None
                    else None
                ),
                options_model=plugin.options_model.__name__ if plugin.options_model else None,
                options_default=plugin.options_default,
                version_hash=plugin.version_hash,
                contract_checkers=len(plugin.contract_checkers),
                config_schema_ref=plugin.config_schema_ref,
                row_count_tables=plugin.row_count_tables,
                cache_populates=plugin.cache_populates,
                cache_consumes=plugin.cache_consumes,
            )
            for plugin in plan.plugins
        }
        resp = GraphPlanResponse(
            plan_id=plan.plan_id,
            ordered_plugins=plan.ordered_names,
            skipped_plugins=tuple(
                GraphPlanSkipped(name=skipped.name, reason=skipped.reason)
                for skipped in plan.skipped_plugins
            ),
            dep_graph={name: tuple(deps) for name, deps in plan.dep_graph.items()},
            plugin_metadata=metadata,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_function_architecture(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: FunctionArchitectureResponse = backend.get_function_architecture(goid_h128=goid_h128)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_module_architecture(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: ModuleArchitectureResponse = backend.get_module_architecture(module=module)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def list_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: SubsystemSummaryResponse = backend.list_subsystems(limit=limit, role=role, q=q)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_module_subsystems(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: ModuleSubsystemResponse = backend.get_module_subsystems(module=module)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_file_hints(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: FileHintsResponse = backend.get_file_hints(rel_path=rel_path)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_subsystem_modules(
        subsystem_id: str, module_limit: int | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: SubsystemModulesResponse = backend.get_subsystem_modules(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def search_subsystems(
        limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: SubsystemSearchResponse = backend.search_subsystems(limit=limit, role=role, q=q)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def summarize_subsystem(
        subsystem_id: str, module_limit: int | None = None
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: SubsystemModulesResponse = backend.summarize_subsystem(
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )
        return resp.model_dump()


__all__ = ["register_architecture_tools"]
