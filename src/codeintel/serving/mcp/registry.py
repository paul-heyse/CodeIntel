"""MCP tool registration and error-to-problem mapping."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from codeintel.analytics.graphs.plugins import plan_graph_metric_plugins
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.backend import QueryBackend
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetRowsResponse,
    FileHintsResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionSummaryResponse,
    GraphPlanPluginMetadata,
    GraphPlanResponse,
    GraphPlanSkipped,
    HighRiskFunctionsResponse,
    ModuleArchitectureResponse,
    ModuleSubsystemResponse,
    ProblemDetail,
    SubsystemModulesResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.serving.services.query_service import QueryService


def _wrap(tool: Callable[..., object]) -> Callable[..., object]:
    """
    Wrap a backend-facing tool to normalize McpError into ProblemDetail payloads.

    Returns
    -------
    Callable[..., object]
        Wrapped tool function that emits dict payloads.
    """

    def _inner(*args: object, **kwargs: object) -> object:
        try:
            return tool(*args, **kwargs)
        except errors.McpError as exc:
            return {"error": exc.detail.model_dump()}

    return _inner


def _register_function_tools(mcp: FastMCP, backend: QueryBackend | QueryService) -> None:
    """Register function-centric MCP tools."""

    @mcp.tool()
    @_wrap
    def get_function_summary(
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: FunctionSummaryResponse = backend.get_function_summary(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def list_high_risk_functions(
        min_risk: float = 0.7,
        limit: int = 50,
        *,
        tested_only: bool = False,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: HighRiskFunctionsResponse = backend.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_callgraph_neighbors(
        goid_h128: int,
        direction: str = "both",
        limit: int = 50,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: CallGraphNeighborsResponse = backend.get_callgraph_neighbors(
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_tests_for_function(
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: TestsForFunctionResponse = backend.get_tests_for_function(
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
        )
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_file_summary(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: FileSummaryResponse = backend.get_file_summary(rel_path=rel_path)
        return resp.model_dump()


def _register_profile_tools(mcp: FastMCP, backend: QueryBackend | QueryService) -> None:
    """Register profile-oriented MCP tools."""

    @mcp.tool()
    @_wrap
    def get_function_profile(goid_h128: int) -> dict[str, object] | dict[str, ProblemDetail]:
        resp = backend.get_function_profile(goid_h128=goid_h128)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_file_profile(rel_path: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp = backend.get_file_profile(rel_path=rel_path)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def get_module_profile(module: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp = backend.get_module_profile(module=module)
        return resp.model_dump()


def _register_architecture_tools(mcp: FastMCP, backend: QueryBackend | QueryService) -> None:
    """Register architecture and subsystem MCP tools."""

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
        names:
            Explicit plugin names to plan (used when enable is not provided).
        enable:
            Ordered list of plugins to enable (overrides defaults when provided).
        disable:
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
        limit: int = 50, role: str | None = None, q: str | None = None
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
    def get_subsystem_modules(subsystem_id: str) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: SubsystemModulesResponse = backend.get_subsystem_modules(subsystem_id=subsystem_id)
        return resp.model_dump()

    @mcp.tool()
    @_wrap
    def search_subsystems(
        limit: int = 20, role: str | None = None, q: str | None = None
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


def _register_dataset_tools(mcp: FastMCP, backend: QueryBackend | QueryService) -> None:
    """Register dataset browsing MCP tools."""

    @mcp.tool()
    @_wrap
    def list_datasets() -> list[dict[str, object]]:
        return [descriptor.model_dump() for descriptor in backend.list_datasets()]

    @mcp.tool()
    @_wrap
    def read_dataset_rows(
        dataset_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, object] | dict[str, ProblemDetail]:
        resp: DatasetRowsResponse = backend.read_dataset_rows(
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )
        return resp.model_dump()


def register_tools(mcp: FastMCP, backend: QueryBackend | QueryService) -> None:
    """
    Register all MCP tools on the given FastMCP instance.

    Parameters
    ----------
    mcp:
        FastMCP instance to register tools against.
    backend:
        Concrete MCP backend or any QueryService implementation.
    """
    _register_function_tools(mcp, backend)
    _register_profile_tools(mcp, backend)
    _register_architecture_tools(mcp, backend)
    _register_dataset_tools(mcp, backend)
