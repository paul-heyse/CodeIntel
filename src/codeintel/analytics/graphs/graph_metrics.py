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

from codeintel.analytics.compute.graphs import (
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
from codeintel.analytics.compute.row_builders import (
    FunctionGraphMetricInputs,
    ModuleGraphMetricInputs,
    build_function_graph_metric_rows,
    build_module_graph_metric_rows,
    component_metadata_from_import_table,
    load_symbol_module_edges,
    merge_component_metadata,
)
from codeintel.graphs.runtime import (
    GraphMetricsOptions,
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.modules import ModuleRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsGraphMetricsFunctionsRow as GraphMetricsFunctionsRow,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsGraphMetricsModulesRow as GraphMetricsModulesRow,
    )
    from codeintel.graphs.runtime.context import GraphContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphMetricsDeps:
    """Optional dependencies for graph metrics execution."""

    catalog_provider: FunctionCatalogProvider | None = None
    runtime: GraphRuntime | GraphRuntimeOptions | None = None
    filters: GraphMetricFilters | None = None
    module_by_path: dict[str, str] | None = None


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
        allowed = self.function_goids
        present = tuple(node for node in allowed if node in graph)
        filtered = nx.DiGraph()
        filtered.add_nodes_from(present)
        filtered.add_edges_from(
            (node, nbr) for node in present for nbr in graph.successors(node) if nbr in allowed
        )
        return filtered

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


@dataclass(frozen=True)
class GraphMetricsRows:
    """Rows for function and module graph metrics."""

    function_rows: list[GraphMetricsFunctionsRow]
    module_rows: list[GraphMetricsModulesRow]


def build_graph_metric_filters(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> GraphMetricFilters:
    """
    Construct repository-backed filters for graph metrics.

    When repositories return no data, filters default to no-ops.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).

    Returns
    -------
    GraphMetricFilters
        Filter set derived from repository contents.
    """
    func_repo = FunctionRepository(gateway=gateway, repo=snapshot.repo, commit=snapshot.commit)
    module_repo = ModuleRepository(gateway=gateway, repo=snapshot.repo, commit=snapshot.commit)
    function_goids = set(func_repo.list_function_goids())
    modules = set(module_repo.list_modules())
    subsystem_repo = SubsystemRepository(
        gateway=gateway, repo=snapshot.repo, commit=snapshot.commit
    )
    subsystem_ids: set[str] = set()
    for row in subsystem_repo.list_subsystem_memberships():
        subsystem_id = row.get("subsystem_id")
        if isinstance(subsystem_id, str):
            subsystem_ids.add(subsystem_id)
    return GraphMetricFilters(
        function_goids=function_goids or None,
        modules=modules or None,
        subsystems=subsystem_ids or None,
    )


def build_graph_metrics_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: GraphMetricsOptions | None = None,
    deps: GraphMetricsDeps | None = None,
) -> GraphMetricsRows:
    """
    Populate analytics graph metrics tables for the provided repo/commit.

    Parameters
    ----------
    gateway
        Storage gateway used for graph reads and metric writes.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).
    options
        Graph metrics configuration options.
    deps
        Optional dependencies container (catalog_provider, runtime, filters, module_by_path).

    Returns
    -------
    GraphMetricsRows
        Row bundles for graph metrics tables.
    """
    opts = options or GraphMetricsOptions()
    deps = deps or GraphMetricsDeps()
    catalog_provider = deps.catalog_provider
    runtime = deps.runtime
    runtime_opts: GraphRuntimeOptions = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    runtime_input: GraphRuntime | GraphRuntimeOptions = (
        runtime if runtime is not None else runtime_opts
    )
    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_input,
    )
    use_gpu = resolved_runtime.backend.use_gpu

    ctx = resolve_graph_context(
        GraphContextSpec(
            repo=snapshot.repo,
            commit=snapshot.commit,
            use_gpu=use_gpu,
            options=opts,
            now=datetime.now(tz=UTC),
            community_detection_limit=runtime_opts.features.community_detection_limit,
        )
    )
    active_filters = deps.filters or build_graph_metric_filters(gateway, snapshot)
    log.info(
        "graph_metrics.filters repo=%s commit=%s functions=%d modules=%d subsystems=%d",
        snapshot.repo,
        snapshot.commit,
        len(active_filters.function_goids or ()),
        len(active_filters.modules or ()),
        len(active_filters.subsystems or ()),
    )
    function_rows = _build_function_graph_metrics_rows(
        gateway, snapshot, ctx=ctx, runtime=resolved_runtime, filters=active_filters
    )
    module_by_path = deps.module_by_path
    if module_by_path is None and catalog_provider is not None:
        module_by_path = catalog_provider.catalog().module_by_path
    module_options = ModuleMetricOptions(module_by_path=module_by_path, filters=active_filters)
    module_rows = _build_module_graph_metrics_rows(
        gateway, snapshot, ctx=ctx, runtime=resolved_runtime, options=module_options
    )
    return GraphMetricsRows(function_rows=function_rows, module_rows=module_rows)


def _build_function_graph_metrics_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    ctx: GraphContext,
    runtime: GraphRuntime,
    filters: GraphMetricFilters,
) -> list[GraphMetricsFunctionsRow]:
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

    rows = build_function_graph_metric_rows(
        FunctionGraphMetricInputs(
            repo=snapshot.repo,
            commit=snapshot.commit,
            stats=stats,
            centrality=centrality,
            components=components,
            graph_nodes=sorted(graph.nodes),
            created_at=created_at,
        )
    )

    if rows:
        log.info(
            "graph_metrics_functions rows built: %d rows for %s@%s",
            len(rows),
            snapshot.repo,
            snapshot.commit,
        )
    return rows


def _build_module_graph_metrics_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    ctx: GraphContext,
    runtime: GraphRuntime,
    options: ModuleMetricOptions,
) -> list[GraphMetricsModulesRow]:
    filters = options.filters or GraphMetricFilters()
    graph = filters.filter_import_graph(runtime.ensure_import_graph())
    symbol_edges = load_symbol_module_edges(gateway, options.module_by_path)
    modules = set(graph.nodes) | symbol_edges[0]
    modules.update(
        ModuleRepository(gateway=gateway, repo=snapshot.repo, commit=snapshot.commit).list_modules()
    )
    if filters.modules is not None:
        modules = modules.intersection(filters.modules)
    if modules:
        graph.add_nodes_from(modules)

    import_stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, ctx)
    component_raw = component_metadata(graph)
    cached_component_meta = component_metadata_from_import_table(
        gateway, snapshot.repo, snapshot.commit
    )
    component_meta = merge_component_metadata(
        modules,
        {
            "component_id": dict(component_raw.component_id),
            "in_cycle": dict(component_raw.in_cycle),
            "layer": dict(component_raw.layer),
        },
        cached_component_meta,
    )

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }
    rows_to_insert = build_module_graph_metric_rows(
        ModuleGraphMetricInputs(
            repo=snapshot.repo,
            commit=snapshot.commit,
            modules=modules,
            import_stats=import_stats,
            centrality=centrality,
            component_meta=component_meta,
            symbol_inbound=symbol_edges[1],
            symbol_outbound=symbol_edges[2],
            created_at=ctx.resolved_now(),
        )
    )

    if rows_to_insert:
        log.info(
            "graph_metrics_modules rows built: %d rows for %s@%s",
            len(rows_to_insert),
            snapshot.repo,
            snapshot.commit,
        )
    return rows_to_insert
