"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural hotspots and coupling signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.analytics.graph_rows import (
    FunctionGraphMetricInputs,
    ModuleGraphMetricInputs,
    build_function_graph_metric_rows,
    build_module_graph_metric_rows,
    component_metadata_from_import_table,
    load_symbol_module_edges,
    merge_component_metadata,
)
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
from codeintel.analytics.graph_service_runtime import GraphContextSpec, resolve_graph_context
from codeintel.config import GraphMetricsStepConfig
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from codeintel.storage.sql_helpers import ensure_schema

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphMetricsDeps:
    """Optional dependencies for graph metrics execution."""

    catalog_provider: FunctionCatalogProvider | None = None
    runtime: GraphRuntime | GraphRuntimeOptions | None = None
    analytics_context: AnalyticsContext | None = None
    filters: GraphMetricFilters | None = None


@dataclass(frozen=True)
class ModuleMetricOptions:
    """Options for module graph metric computation."""

    module_by_path: dict[str, str] | None = None
    filters: GraphMetricFilters | None = None


@dataclass(frozen=True)
class GraphMetricFilters:
    """Optional filters for graph metric node sets."""

    function_goids: set[int] | None = None
    modules: set[str] | None = None
    subsystems: set[str] | None = None

    def filter_call_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Return a filtered call graph when a function allowlist is provided.

        Returns
        -------
        nx.DiGraph
            Subgraph restricted to allowed GOIDs or the original graph.
        """
        if not self.function_goids:
            return graph
        return nx.subgraph(graph, self.function_goids).copy()

    def filter_import_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Return a filtered import graph when a module allowlist is provided.

        Returns
        -------
        nx.DiGraph
            Subgraph restricted to allowed modules or the original graph.
        """
        if not self.modules:
            return graph
        return nx.subgraph(graph, self.modules).copy()

    def filter_subsystem_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Return a filtered subsystem graph when an allowlist is provided.

        Returns
        -------
        nx.DiGraph
            Subgraph restricted to allowed subsystem ids or the original graph.
        """
        if not self.subsystems:
            return graph
        return nx.subgraph(graph, self.subsystems).copy()

    def filter_subsystem_memberships(
        self, memberships: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """
        Filter subsystem-module memberships using subsystem and module allowlists.

        Returns
        -------
        list[tuple[str, str]]
            Filtered membership rows.
        """
        if not self.subsystems and not self.modules:
            return memberships
        return [
            (subsystem_id, module)
            for subsystem_id, module in memberships
            if (not self.subsystems or subsystem_id in self.subsystems)
            and (not self.modules or module in self.modules)
        ]


def build_graph_metric_filters(
    gateway: StorageGateway, cfg: GraphMetricsStepConfig
) -> GraphMetricFilters:
    """
    Construct repository-backed filters for graph metrics.

    When repositories return no data, filters default to no-ops.

    Returns
    -------
    GraphMetricFilters
        Filter set derived from repository contents.
    """
    func_repo = FunctionRepository(gateway=gateway, repo=cfg.repo, commit=cfg.commit)
    module_repo = ModuleRepository(gateway=gateway, repo=cfg.repo, commit=cfg.commit)
    function_goids = set(func_repo.list_function_goids())
    modules = set(module_repo.list_modules())
    subsystem_repo = SubsystemRepository(gateway=gateway, repo=cfg.repo, commit=cfg.commit)
    subsystem_ids = {row["subsystem_id"] for row in subsystem_repo.list_subsystem_memberships()}
    return GraphMetricFilters(
        function_goids=function_goids or None,
        modules=modules or None,
        subsystems=subsystem_ids or None,
    )


def compute_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    deps: GraphMetricsDeps | None = None,
) -> None:
    """
    Populate analytics graph metrics tables for the provided repo/commit.

    Parameters
    ----------
    gateway :
        Storage gateway used for graph reads and metric writes.
    cfg :
        Graph metrics configuration for the current repository snapshot.
    deps :
        Optional dependencies container (catalog_provider, runtime, analytics_context, filters).
    """
    deps = deps or GraphMetricsDeps()
    catalog_provider = deps.catalog_provider
    runtime = deps.runtime
    analytics_context = deps.analytics_context
    runtime_opts: GraphRuntimeOptions = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    runtime_input: GraphRuntime | GraphRuntimeOptions = (
        runtime if runtime is not None else runtime_opts
    )
    resolved_runtime = resolve_graph_runtime(
        gateway,
        cfg.snapshot,
        runtime_input,
        context=analytics_context or runtime_opts.context,
    )
    use_gpu = resolved_runtime.backend.use_gpu

    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions")
    ensure_schema(con, "analytics.graph_metrics_modules")
    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=cfg.repo,
            commit=cfg.commit,
            use_gpu=use_gpu,
            metrics_cfg=cfg,
            now=datetime.now(tz=UTC),
            community_detection_limit=runtime_opts.features.community_detection_limit,
        )
    )
    active_filters = deps.filters or build_graph_metric_filters(gateway, cfg)
    log.info(
        "graph_metrics.filters repo=%s commit=%s functions=%d modules=%d subsystems=%d",
        cfg.repo,
        cfg.commit,
        len(active_filters.function_goids or ()),
        len(active_filters.modules or ()),
        len(active_filters.subsystems or ()),
    )
    _compute_function_graph_metrics(
        gateway, cfg, ctx=ctx, runtime=resolved_runtime, filters=active_filters
    )
    module_by_path = None
    active_context = analytics_context or runtime_opts.context
    if active_context is not None:
        module_by_path = active_context.module_map
    elif catalog_provider is not None:
        module_by_path = catalog_provider.catalog().module_by_path
    module_options = ModuleMetricOptions(module_by_path=module_by_path, filters=active_filters)
    _compute_module_graph_metrics(
        gateway, cfg, ctx=ctx, runtime=resolved_runtime, options=module_options
    )


def _compute_function_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    ctx: GraphContext,
    runtime: GraphRuntime,
    filters: GraphMetricFilters,
) -> None:
    con = gateway.con
    graph = filters.filter_call_graph(runtime.ensure_call_graph())
    stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, ctx)
    components = component_metadata(graph)
    created_at = ctx.resolved_now()

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }

    con.execute(
        "DELETE FROM analytics.graph_metrics_functions WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    rows = build_function_graph_metric_rows(
        FunctionGraphMetricInputs(
            cfg=cfg,
            stats=stats,
            centrality=centrality,
            components=components,
            graph_nodes=sorted(graph.nodes),
            created_at=created_at,
        )
    )

    if rows:
        gateway.analytics.insert_graph_metrics_functions(rows)
        log.info(
            "graph_metrics_functions populated: %d rows for %s@%s",
            len(rows),
            cfg.repo,
            cfg.commit,
        )


def _compute_module_graph_metrics(
    gateway: StorageGateway,
    cfg: GraphMetricsStepConfig,
    *,
    ctx: GraphContext,
    runtime: GraphRuntime,
    options: ModuleMetricOptions,
) -> None:
    con = gateway.con
    filters = options.filters or GraphMetricFilters()
    graph = filters.filter_import_graph(runtime.ensure_import_graph())
    symbol_modules, symbol_inbound, symbol_outbound = load_symbol_module_edges(
        gateway, options.module_by_path
    )
    modules = set(graph.nodes) | symbol_modules
    module_repo = ModuleRepository(gateway=gateway, repo=cfg.repo, commit=cfg.commit)
    modules.update(module_repo.list_modules())
    if filters.modules is not None:
        modules = modules.intersection(filters.modules)
    if modules:
        graph.add_nodes_from(modules)

    import_stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, ctx)
    component_raw = component_metadata(graph)
    cached_component_meta = component_metadata_from_import_table(gateway, cfg.repo, cfg.commit)
    component_meta = merge_component_metadata(
        modules,
        {
            "component_id": dict(component_raw.component_id),
            "in_cycle": dict(component_raw.in_cycle),
            "layer": dict(component_raw.layer),
        },
        cached_component_meta,
    )

    con.execute(
        "DELETE FROM analytics.graph_metrics_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }
    rows_to_insert = build_module_graph_metric_rows(
        ModuleGraphMetricInputs(
            cfg=cfg,
            modules=modules,
            import_stats=import_stats,
            centrality=centrality,
            component_meta=component_meta,
            symbol_inbound=symbol_inbound,
            symbol_outbound=symbol_outbound,
            created_at=ctx.resolved_now(),
        )
    )

    if rows_to_insert:
        gateway.analytics.insert_graph_metrics_modules(rows_to_insert)
        log.info(
            "graph_metrics_modules populated: %d rows for %s@%s",
            len(rows_to_insert),
            cfg.repo,
            cfg.commit,
        )
