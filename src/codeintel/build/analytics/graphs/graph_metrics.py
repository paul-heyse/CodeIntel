"""
Compute graph-theoretic metrics for functions and modules.

This module derives call-graph and import-graph metrics that help surface
architectural bottlenecks and coupling signals.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

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
)
from codeintel.build.analytics.graphs.orchestrator import (
    MetricsPipelineConfig,
    MetricsPipelineRequest,
    build_graph_views,
    build_metrics_pipeline_rows,
)
from codeintel.build.graphs.builders import (
    build_call_graph_from_rows as _build_call_graph_from_rows,
)
from codeintel.build.graphs.builders import (
    build_call_graph_from_tables as _build_call_graph_from_tables,
)
from codeintel.build.graphs.builders import (
    build_import_graph_from_rows as _build_import_graph_from_rows,
)
from codeintel.build.graphs.builders import (
    build_import_graph_from_tables as _build_import_graph_from_tables,
)
from codeintel.build.graphs.external_plan import run_rustworkx_external_plan
from codeintel.build.graphs.runtime import GraphMetricsOptions, GraphRuntimeOptions
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.analytics.compute.graphs import ComponentBundle, NeighborStats
    from codeintel.build.analytics.graphs.orchestrator import GraphViews
    from codeintel.build.graphs.runtime.context import GraphContext
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

SymbolModuleEdges = tuple[set[str], dict[str, set[str]], dict[str, set[str]]]
ComponentMeta = Mapping[str, Mapping[str, int | bool]]
GraphMetricRowSource = ColumnarRowBuffer | pa.RecordBatchReader | pa.Table


def _filter_store(graph: GraphInput, allowed: Collection[Hashable]) -> RxGraphStore:
    store = ensure_store(graph)
    allowed_set = set(allowed)
    node_indices = [
        store.id_to_index[node_id] for node_id in allowed_set if node_id in store.id_to_index
    ]
    if not node_indices:
        if store.is_directed:
            return RxGraphStore.directed(
                weight_policy=store.weight_policy,
                numeric_policy=store.numeric_policy,
            )
        return RxGraphStore.undirected(
            weight_policy=store.weight_policy,
            numeric_policy=store.numeric_policy,
        )
    node_indices.sort(key=lambda idx: stable_key(store.index_to_id[idx]))
    subgraph, _ = store.graph.subgraph_with_nodemap(node_indices, preserve_attrs=True)
    return RxGraphStore.from_rx_graph(
        subgraph,
        weight_policy=store.weight_policy,
        numeric_policy=store.numeric_policy,
    )


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

    function_rows: GraphMetricRowSource
    module_rows: GraphMetricRowSource


@dataclass(frozen=True)
class FunctionMetricSlices:
    """Precomputed graph statistics for function metrics rows."""

    stats: NeighborStats
    centrality: Mapping[str, Mapping[int, float]]
    components: ComponentBundle
    graph_nodes: list[int]


@dataclass(frozen=True)
class ModuleMetricSlices:
    """Precomputed graph statistics for module metrics rows."""

    modules: set[str]
    import_stats: NeighborStats
    centrality: Mapping[str, Mapping[str, float]]
    component_meta: Mapping[str, Mapping[str, int | bool]]
    symbol_inbound: Mapping[str, set[str]]
    symbol_outbound: Mapping[str, set[str]]


@dataclass(frozen=True)
class _ModuleMetricSliceInputs:
    """Inputs required to compute module metric slices."""

    symbol_module_edges: SymbolModuleEdges
    module_names: Iterable[str]
    component_meta_cache: ComponentMeta | None
    filters: GraphMetricFilters


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


def build_call_graph_from_tables(
    call_graph_edges: pa.Table,
    call_graph_nodes: pa.Table | None = None,
    *,
    repo: str | None = None,
    commit: str | None = None,
) -> GraphInput:
    """Build a call graph from Arrow tables with Acero aggregation.

    Returns
    -------
    GraphInput
        Directed call graph populated from the provided tables.
    """
    return _build_call_graph_from_tables(
        call_graph_edges,
        call_graph_nodes,
        repo=repo,
        commit=commit,
    )


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


def build_import_graph_from_tables(
    import_graph_edges: pa.Table,
    import_modules: pa.Table | None = None,
    *,
    repo: str | None = None,
    commit: str | None = None,
) -> GraphInput:
    """Build an import graph from Arrow tables with Acero aggregation.

    Returns
    -------
    GraphInput
        Directed import graph populated from the provided tables.
    """
    return _build_import_graph_from_tables(
        import_graph_edges,
        import_modules,
        repo=repo,
        commit=commit,
    )


@dataclass(frozen=True)
class _GraphMetricsContextBundle:
    runtime: GraphRuntimeOptions
    filters: GraphMetricFilters
    overrides: GraphContextOverrides

    def build_context(
        self,
        runtime_opts: GraphRuntimeOptions,
        repo: str,
        commit: str,
    ) -> GraphContext:
        return _GRAPH_CONTEXT_FACTORY.build(
            runtime_opts,
            repo=repo,
            commit=commit,
            overrides=self.overrides,
        )


def _resolve_graph_metrics_context(inputs: GraphMetricsInputs) -> _GraphMetricsContextBundle:
    opts = inputs.options or GraphMetricsOptions()
    runtime = GraphRuntimeOptions(
        snapshot=inputs.snapshot,
        backend=GraphBackendConfig(use_gpu=inputs.use_gpu),
        features=GraphFeatureFlags(community_detection_limit=inputs.community_detection_limit),
    )
    filters = inputs.filters or GraphMetricFilters()
    log.info(
        "graph_metrics.filters repo=%s commit=%s functions=%d modules=%d subsystems=%d",
        inputs.snapshot.repo,
        inputs.snapshot.commit,
        len(filters.function_goids or ()),
        len(filters.modules or ()),
        len(filters.subsystems or ()),
    )
    overrides = GraphContextOverrides(
        options=opts,
        use_gpu=inputs.use_gpu,
        community_detection_limit=inputs.community_detection_limit,
    )
    return _GraphMetricsContextBundle(
        runtime=runtime,
        filters=filters,
        overrides=overrides,
    )


def build_graph_metrics_function_rows(
    inputs: GraphMetricsInputs,
) -> ColumnarRowBuffer:
    """Build function graph metrics rows for analytics outputs.

    Parameters
    ----------
    inputs
        Structured graph metric inputs (graphs, filters, options, and snapshot).

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing function-level graph metrics rows.
    """
    context = _resolve_graph_metrics_context(inputs)
    config = MetricsPipelineConfig(
        table_key="analytics.graph_metrics_functions",
        filter_graph=lambda filters, graph: filters.filter_call_graph(graph),
        build_context=context.build_context,
        build_views=build_graph_views,
        build_slices=_function_metric_slices,
        build_rows=_function_metric_rows,
    )
    request = MetricsPipelineRequest(
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        graph=inputs.call_graph,
        runtime=context.runtime,
        filters=context.filters,
    )
    return build_metrics_pipeline_rows(config, request)


def build_graph_metrics_module_rows(
    inputs: GraphMetricsInputs,
) -> ColumnarRowBuffer:
    """Build module graph metrics rows for analytics outputs.

    Parameters
    ----------
    inputs
        Structured graph metric inputs (graphs, filters, options, and snapshot).

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing module-level graph metrics rows.
    """
    context = _resolve_graph_metrics_context(inputs)
    module_slice_inputs = _ModuleMetricSliceInputs(
        symbol_module_edges=inputs.symbol_module_edges,
        module_names=inputs.module_names,
        component_meta_cache=inputs.component_meta,
        filters=context.filters,
    )

    def _module_slices(views: GraphViews, ctx: GraphContext) -> ModuleMetricSlices:
        return _module_metric_slices(views, ctx, module_slice_inputs)

    config = MetricsPipelineConfig(
        table_key="analytics.graph_metrics_modules",
        filter_graph=lambda filters, graph: filters.filter_import_graph(graph),
        build_context=context.build_context,
        build_views=build_graph_views,
        build_slices=_module_slices,
        build_rows=_module_metric_rows,
    )
    request = MetricsPipelineRequest(
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        graph=inputs.import_graph,
        runtime=context.runtime,
        filters=context.filters,
    )
    return build_metrics_pipeline_rows(config, request)


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
    function_rows = build_graph_metrics_function_rows(inputs)
    module_rows = build_graph_metrics_module_rows(inputs)
    return GraphMetricsRows(function_rows=function_rows, module_rows=module_rows)


def build_graph_metrics_readers(
    inputs: GraphMetricsInputs,
    *,
    use_threads: bool | None = None,
) -> GraphMetricsRows:
    """Execute graph metrics via the rustworkx external plan runner.

    Parameters
    ----------
    inputs
        Structured graph metric inputs (graphs, filters, options, and snapshot).
    use_threads
        Whether to enable threaded execution for rustworkx plans.

    Returns
    -------
    GraphMetricsRows
        Row sources for graph metrics tables.
    """
    function_reader = run_rustworkx_external_plan(
        builder=build_graph_metrics_function_rows,
        args=(inputs,),
        use_threads=use_threads,
    )
    module_reader = run_rustworkx_external_plan(
        builder=build_graph_metrics_module_rows,
        args=(inputs,),
        use_threads=use_threads,
    )
    return GraphMetricsRows(
        function_rows=function_reader,
        module_rows=module_reader,
    )


def _function_metric_slices(
    views: GraphViews,
    ctx: GraphContext,
) -> FunctionMetricSlices:
    stats = neighbor_stats(views.graph, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(views.graph, ctx)
    components = component_metadata(views.graph)
    centrality = {
        "pagerank": centrality_bundle.pagerank,
        "betweenness": centrality_bundle.betweenness,
        "closeness": centrality_bundle.closeness,
    }
    graph_nodes: list[int] = []
    for node in views.graph.node_ids():
        node_id = normalize_decimal_id(node)
        if node_id is None:
            continue
        graph_nodes.append(node_id)
    return FunctionMetricSlices(
        stats=stats,
        centrality=centrality,
        components=components,
        graph_nodes=graph_nodes,
    )


def _function_metric_rows(
    repo: str,
    commit: str,
    ctx: GraphContext,
    _views: GraphViews,
    slices: FunctionMetricSlices,
) -> ColumnarRowBuffer:
    row_context = RowBuildContext.from_repo_commit(repo, commit, created_at=ctx.resolved_now())
    rows = build_function_graph_metric_rows(
        FunctionGraphMetricInputs(
            row_context=row_context,
            stats=slices.stats,
            centrality=slices.centrality,
            components=slices.components,
            graph_nodes=slices.graph_nodes,
        )
    )
    if rows:
        log.info(
            "graph_metrics_functions rows built: %d rows for %s@%s",
            rows.row_count,
            row_context.repo,
            row_context.commit,
        )
    return rows


def _module_metric_slices(
    views: GraphViews,
    ctx: GraphContext,
    inputs: _ModuleMetricSliceInputs,
) -> ModuleMetricSlices:
    graph_store = views.graph
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

    import_stats = neighbor_stats(graph_store, weight=ctx.betweenness_weight)
    centrality_bundle = centrality_directed(graph_store, ctx)
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
    return ModuleMetricSlices(
        modules=modules,
        import_stats=import_stats,
        centrality=centrality,
        component_meta=component_meta,
        symbol_inbound=symbol_inbound,
        symbol_outbound=symbol_outbound,
    )


def _module_metric_rows(
    repo: str,
    commit: str,
    ctx: GraphContext,
    _views: GraphViews,
    slices: ModuleMetricSlices,
) -> ColumnarRowBuffer:
    row_context = RowBuildContext.from_repo_commit(repo, commit, created_at=ctx.resolved_now())
    rows = build_module_graph_metric_rows(
        ModuleGraphMetricInputs(
            row_context=row_context,
            modules=slices.modules,
            import_stats=slices.import_stats,
            centrality=slices.centrality,
            component_meta=slices.component_meta,
            symbol_inbound=slices.symbol_inbound,
            symbol_outbound=slices.symbol_outbound,
        )
    )
    if rows:
        log.info(
            "graph_metrics_modules rows built: %d rows for %s@%s",
            rows.row_count,
            row_context.repo,
            row_context.commit,
        )
    return rows


_GRAPH_CONTEXT_FACTORY = GraphContextFactory()
