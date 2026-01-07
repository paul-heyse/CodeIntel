"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural bottlenecks and coupling signals.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    GraphContextOverrides,
    GraphMetricsContext,
)
from codeintel.build.graphs.builders import (
    build_call_graph_from_rows as _build_call_graph_from_rows,
)
from codeintel.build.graphs.builders import (
    build_import_graph_from_rows as _build_import_graph_from_rows,
)
from codeintel.build.graphs.runtime import GraphMetricsOptions, GraphRuntimeOptions
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.graphs.runtime.context import GraphContext
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

SymbolModuleEdges = tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
ComponentMeta = Mapping[str, Mapping[str, int | bool]]


def _filter_store(graph: GraphInput, allowed: set[Hashable]) -> RxGraphStore:
    store = ensure_store(graph)
    if store.is_directed:
        filtered = RxGraphStore.directed(
            node_hint=store.graph.num_nodes(),
            edge_hint=store.graph.num_edges(),
        )
    else:
        filtered = RxGraphStore.undirected(
            node_hint=store.graph.num_nodes(),
            edge_hint=store.graph.num_edges(),
        )
    allowed_set = set(allowed)
    for node_id in store.node_ids():
        if node_id in allowed_set:
            filtered.set_node_attrs(node_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        if src_id not in allowed_set or dst_id not in allowed_set:
            continue
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        filtered.add_weighted_edge(src_id, dst_id, weight=weight)
    return filtered


@dataclass(frozen=True)
class GraphMetricFilters:
    """Optional filters for graph metric node sets."""

    function_goids: set[Hashable] | None = None
    modules: set[str] | None = None
    subsystems: set[str] | None = None

    def filter_call_graph(self, graph: GraphInput) -> GraphInput:
        """
        Return a filtered call graph when a function allowlist is provided.

        Returns
        -------
        GraphInput
            Subgraph restricted to allowed GOIDs or the original graph.
        """
        if not self.function_goids:
            return ensure_store(graph)
        return _filter_store(graph, self.function_goids)

    def filter_import_graph(self, graph: GraphInput) -> GraphInput:
        """
        Return a filtered import graph when a module allowlist is provided.

        Returns
        -------
        GraphInput
            Subgraph restricted to allowed modules or the original graph.
        """
        if not self.modules:
            return ensure_store(graph)
        return _filter_store(graph, self.modules)

    def filter_subsystem_graph(self, graph: GraphInput) -> GraphInput:
        """
        Return a filtered subsystem graph when an allowlist is provided.

        Returns
        -------
        GraphInput
            Subgraph restricted to allowed subsystem ids or the original graph.
        """
        if not self.subsystems:
            return ensure_store(graph)
        return _filter_store(graph, self.subsystems)

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
    call_graph: GraphInput
    import_graph: GraphInput
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
    import_graph: GraphInput
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
) -> GraphInput:
    """Build a call graph from scoped call graph edge/node rows.

    Returns
    -------
    GraphInput
        Directed call graph populated from the provided rows.
    """
    return _build_call_graph_from_rows(call_graph_edges, call_graph_nodes)


def build_import_graph_from_rows(
    import_graph_edges: Iterable[Mapping[str, object]],
    import_modules: Iterable[Mapping[str, object]] | None = None,
) -> GraphInput:
    """Build an import graph from scoped import edges and module rows.

    Returns
    -------
    GraphInput
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
        filters=inputs.filters or GraphMetricFilters(),
        context_factory=_GRAPH_CONTEXT_FACTORY,
        overrides=GraphContextOverrides(
            options=opts,
            use_gpu=inputs.use_gpu,
            community_detection_limit=inputs.community_detection_limit,
        ),
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
    *,
    ctx: GraphContext,
    call_graph: GraphInput,
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

    store = ensure_store(graph)
    graph_nodes: list[int] = []
    for node in store.node_ids():
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
            row_context.repo,
            row_context.commit,
        )
    return rows


def _build_module_graph_metrics_rows(
    inputs: ModuleGraphMetricsInputs,
) -> list[dict[str, object]]:
    graph = inputs.filters.filter_import_graph(inputs.import_graph)
    graph_store = ensure_store(graph)
    symbol_modules, symbol_inbound, symbol_outbound = inputs.symbol_module_edges
    modules = {str(node) for node in graph_store.node_ids()} | {
        str(node) for node in symbol_modules
    }
    modules.update(str(module) for module in inputs.module_names)
    if inputs.filters.modules is not None:
        modules = modules.intersection(inputs.filters.modules)
    if modules:
        for module in modules:
            graph_store.ensure_node(module)

    import_stats = neighbor_stats(graph_store, weight=inputs.ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph_store, inputs.ctx)
    component_raw = component_metadata(graph_store)
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
