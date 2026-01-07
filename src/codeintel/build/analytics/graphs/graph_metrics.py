"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural bottlenecks and coupling signals.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import networkx as nx

from codeintel.build.analytics.compute.graphs import (
    centrality_directed,
    component_metadata,
    neighbor_stats,
)
from codeintel.build.analytics.compute.row_builders import (
    FunctionGraphMetricInputs,
    ModuleGraphMetricInputs,
    RowBuildContext,
    build_function_graph_metric_rows,
    build_module_graph_metric_rows,
    merge_component_metadata,
)
from codeintel.build.analytics.graphs.context_helpers import (
    GraphContextFactory,
    GraphMetricsContext,
)
from codeintel.build.graphs.builders import (
    build_call_graph_from_rows as _build_call_graph_from_rows,
)
from codeintel.build.graphs.builders import (
    build_import_graph_from_rows as _build_import_graph_from_rows,
)
from codeintel.build.graphs.runtime import GraphMetricsOptions, GraphRuntimeOptions
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from codeintel.build.graphs.runtime.context import GraphContext
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

SymbolModuleEdges = tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
ComponentMeta = Mapping[str, Mapping[str, int | bool]]


@dataclass(frozen=True)
class GraphMetricFilters:
    """Optional filters for graph metric node sets."""

    function_goids: set[Hashable] | None = None
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
        return cast("nx.DiGraph", nx.subgraph(graph, self.modules).copy())

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
        return cast("nx.DiGraph", nx.subgraph(graph, self.subsystems).copy())

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

    function_rows: list[dict[str, object]]
    module_rows: list[dict[str, object]]


@dataclass(frozen=True)
class GraphMetricsInputs:
    """Inputs required to compute graph metrics rows."""

    snapshot: SnapshotRef
    call_graph: nx.DiGraph
    import_graph: nx.DiGraph
    symbol_module_edges: SymbolModuleEdges
    module_names: Iterable[str]
    component_meta: ComponentMeta | None = None
    filters: GraphMetricFilters | None = None
    options: GraphMetricsOptions | None = None
    community_detection_limit: int | None = None
    use_gpu: bool = False


@dataclass(frozen=True)
class ModuleGraphMetricsInputs:
    """Inputs required to compute module graph metrics rows."""

    row_context: RowBuildContext
    ctx: GraphContext
    import_graph: nx.DiGraph
    symbol_module_edges: SymbolModuleEdges
    module_names: Iterable[str]
    filters: GraphMetricFilters
    component_meta_cache: ComponentMeta | None = None


def build_graph_metric_filters_from_sets(
    *,
    function_goids: Iterable[Hashable] | None = None,
    modules: Iterable[str] | None = None,
    subsystems: Iterable[str] | None = None,
) -> GraphMetricFilters:
    """Build graph metric filters from optional allowlist inputs.

    Returns
    -------
    GraphMetricFilters
        Filter set derived from provided allowlists.
    """
    function_set = set(function_goids) if function_goids else set()
    module_set = {str(module) for module in modules or ()}
    subsystem_set = {str(subsystem) for subsystem in subsystems or ()}
    return GraphMetricFilters(
        function_goids=function_set or None,
        modules=module_set or None,
        subsystems=subsystem_set or None,
    )


def build_call_graph_from_rows(
    call_graph_edges: Iterable[Mapping[str, object]],
    call_graph_nodes: Iterable[Mapping[str, object]] | None = None,
) -> nx.DiGraph:
    """Build a call graph from scoped call graph edge/node rows.

    Returns
    -------
    nx.DiGraph
        Directed call graph populated from the provided rows.
    """
    return _build_call_graph_from_rows(call_graph_edges, call_graph_nodes)


def build_import_graph_from_rows(
    import_graph_edges: Iterable[Mapping[str, object]],
    import_modules: Iterable[Mapping[str, object]] | None = None,
) -> nx.DiGraph:
    """Build an import graph from scoped import edges and module rows.

    Returns
    -------
    nx.DiGraph
        Directed import graph populated from the provided rows.
    """
    return _build_import_graph_from_rows(import_graph_edges, import_modules)


def build_graph_metrics_rows(
    inputs: GraphMetricsInputs,
) -> GraphMetricsRows:
    """
    Populate analytics graph metrics tables for the provided repo/commit.

    Parameters
    ----------
    inputs
        Structured graph metric inputs (graphs, filters, options, and snapshot).

    Returns
    -------
    GraphMetricsRows
        Row bundles for graph metrics tables.
    """
    opts = inputs.options or GraphMetricsOptions()
    runtime = GraphRuntimeOptions(
        snapshot=inputs.snapshot,
        backend=GraphBackendConfig(use_gpu=inputs.use_gpu),
        features=GraphFeatureFlags(community_detection_limit=inputs.community_detection_limit),
    )
    context = GraphMetricsContext.from_inputs(
        snapshot=inputs.snapshot,
        runtime=runtime,
        filters=inputs.filters,
        context_factory=_GRAPH_CONTEXT_FACTORY,
        options=opts,
        use_gpu=inputs.use_gpu,
        community_detection_limit=inputs.community_detection_limit,
    )
    active_filters = context.filters
    ctx = context.graph_context
    log.info(
        "graph_metrics.filters repo=%s commit=%s functions=%d modules=%d subsystems=%d",
        inputs.snapshot.repo,
        inputs.snapshot.commit,
        len(active_filters.function_goids or ()),
        len(active_filters.modules or ()),
        len(active_filters.subsystems or ()),
    )
    row_context = RowBuildContext.from_snapshot(inputs.snapshot, created_at=ctx.resolved_now())
    function_rows = _build_function_graph_metrics_rows(
        inputs.snapshot,
        ctx=ctx,
        call_graph=inputs.call_graph,
        filters=active_filters,
        row_context=row_context,
    )
    module_rows = _build_module_graph_metrics_rows(
        ModuleGraphMetricsInputs(
            row_context=row_context,
            ctx=ctx,
            import_graph=inputs.import_graph,
            symbol_module_edges=inputs.symbol_module_edges,
            module_names=inputs.module_names,
            filters=active_filters,
            component_meta_cache=inputs.component_meta,
        )
    )
    return GraphMetricsRows(function_rows=function_rows, module_rows=module_rows)


def _build_function_graph_metrics_rows(
    snapshot: SnapshotRef,
    *,
    ctx: GraphContext,
    call_graph: nx.DiGraph,
    filters: GraphMetricFilters,
    row_context: RowBuildContext,
) -> list[dict[str, object]]:
    graph = filters.filter_call_graph(call_graph)
    stats = neighbor_stats(graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, ctx)
    components = component_metadata(graph)

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }

    graph_nodes: list[int] = []
    for node in graph.nodes:
        node_id = normalize_decimal_id(node)
        if node_id is None:
            continue
        graph_nodes.append(node_id)
    graph_nodes.sort()

    rows = build_function_graph_metric_rows(
        FunctionGraphMetricInputs(
            row_context=row_context,
            stats=stats,
            centrality=centrality,
            components=components,
            graph_nodes=graph_nodes,
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
    inputs: ModuleGraphMetricsInputs,
) -> list[dict[str, object]]:
    graph = inputs.filters.filter_import_graph(inputs.import_graph)
    symbol_modules, symbol_inbound, symbol_outbound = inputs.symbol_module_edges
    modules = {str(node) for node in graph.nodes} | {str(node) for node in symbol_modules}
    modules.update(str(module) for module in inputs.module_names)
    if inputs.filters.modules is not None:
        modules = modules.intersection(inputs.filters.modules)
    if modules:
        graph.add_nodes_from(modules)

    import_stats = neighbor_stats(graph, weight=inputs.ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph, inputs.ctx)
    component_raw = component_metadata(graph)
    computed_component_meta: dict[str, dict[str, int | bool]] = {
        "component_id": {str(node): int(val) for node, val in component_raw.component_id.items()},
        "in_cycle": {str(node): bool(flag) for node, flag in component_raw.in_cycle.items()},
        "layer": {str(node): int(val) for node, val in component_raw.layer.items()},
    }
    component_meta = merge_component_metadata(
        modules,
        computed_component_meta,
        inputs.component_meta_cache,
    )

    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }
    rows_to_insert = build_module_graph_metric_rows(
        ModuleGraphMetricInputs(
            row_context=inputs.row_context,
            modules=modules,
            import_stats=import_stats,
            centrality=centrality,
            component_meta=component_meta,
            symbol_inbound=symbol_inbound,
            symbol_outbound=symbol_outbound,
        )
    )

    if rows_to_insert:
        log.info(
            "graph_metrics_modules rows built: %d rows for %s@%s",
            len(rows_to_insert),
            inputs.row_context.repo,
            inputs.row_context.commit,
        )
    return rows_to_insert


_GRAPH_CONTEXT_FACTORY = GraphContextFactory()
