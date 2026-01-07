"""Generic orchestration framework for extended graph metrics computation.

This module provides a reusable orchestration pattern for computing extended
graph metrics. It extracts the common workflow shared between function-level
and module-level extended metrics into a configurable generic function.

The orchestrator handles:
1. Runtime option extraction
2. Filter construction
3. Graph filtering
4. View building (graph, simple_graph, undirected)
5. Slice computation delegation
6. Row building delegation for downstream materialization

Example
-------
```python
from codeintel.build.analytics.graphs.orchestrator import (
    ExtendedMetricsConfig,
    ExtendedMetricsRequest,
    GraphViews,
    build_extended_metrics_rows,
)

config = ExtendedMetricsConfig(
    table_key="analytics.graph_metrics_functions_ext",
    filter_graph=lambda f, g: f.filter_call_graph(g),
    build_context=_resolve_function_context,
    build_slices=_function_metric_slices,
    build_rows=_function_metric_rows,
)

request = ExtendedMetricsRequest(
    repo="org/repo",
    commit="abc123",
    graph=call_graph,
)
build_extended_metrics_rows(config, request)
```
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    ensure_directed_store,
    ensure_store,
    to_undirected_store,
)
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.graphs.runtime.context import GraphContext


class GraphFilterProtocol(Protocol):
    """Protocol describing graph filter behaviors."""

    def filter_call_graph(self, graph: GraphInput) -> GraphInput: ...

    def filter_import_graph(self, graph: GraphInput) -> GraphInput: ...

    def filter_subsystem_graph(self, graph: GraphInput) -> GraphInput: ...

    def filter_subsystem_memberships(
        self,
        memberships: list[tuple[str, str]],
    ) -> list[tuple[str, str]]: ...


@dataclass(frozen=True)
class GraphViews:
    """Graph variants used for graph metrics computation.

    This dataclass holds the three standard graph representations needed
    for computing graph metrics: the original graph, a simplified version
    without self-loops, and an undirected view.

    Attributes
    ----------
    graph
        The original graph (directed or undirected).
    simple_graph
        Graph with self-loops removed.
    undirected
        Undirected view of the graph for structural metrics.
    """

    graph: RxGraphStore
    simple_graph: RxGraphStore
    undirected: RxGraphStore


@dataclass(frozen=True)
class ExtendedMetricsConfig[TSlices, TRow: Mapping[str, object]]:
    """Configuration for extended graph metrics orchestration.

    This dataclass captures all the domain-specific callables needed to
    compute a particular type of extended graph metrics. The orchestrator
    uses these callables to delegate to the appropriate implementations.

    Attributes
    ----------
    table_key
        Target table key for downstream materialization (e.g.,
        "analytics.graph_metrics_functions_ext").
    filter_graph
        Callable to filter the graph using the active filters.
    build_context
        Callable to build the graph context with appropriate constants.
    build_slices
        Callable to compute metric slices from graph views and context.
    build_rows
        Callable to build rows from slices for downstream materialization.
    """

    table_key: str
    filter_graph: Callable[[GraphFilterProtocol, GraphInput], GraphInput]
    build_context: Callable[[GraphRuntimeOptions, str, str], GraphContext]
    build_slices: Callable[[GraphViews, GraphContext], TSlices]
    build_rows: Callable[[str, str, GraphContext, GraphViews, TSlices], list[TRow]]


@dataclass(frozen=True)
class ExtendedMetricsRequest:
    """Request parameters for extended graph metrics computation.

    Bundles the common request parameters needed to compute extended
    graph metrics, simplifying the orchestrator's function signature.

    Attributes
    ----------
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    graph
        Source graph to analyze (call graph or import graph).
    runtime
        Optional runtime options including backend selection.
    filters
        Optional allowlists for restricting graph nodes.
    """

    repo: str
    commit: str
    graph: GraphInput
    runtime: GraphRuntimeOptions | None = None
    filters: GraphFilterProtocol | None = None


@dataclass(frozen=True)
class MetricsPipelineConfig[TSlices, TRow]:
    """Configuration for graph metrics pipelines.

    Attributes
    ----------
    table_key
        Target table key for downstream materialization.
    filter_graph
        Callable to filter the graph using the active filters.
    build_context
        Callable to build the graph context with appropriate constants.
    build_views
        Callable to build graph views from the filtered graph.
    build_slices
        Callable to compute metric slices from graph views and context.
    build_rows
        Callable to build rows from slices for downstream materialization.
    """

    table_key: str
    filter_graph: Callable[[GraphFilterProtocol, GraphInput], GraphInput]
    build_context: Callable[[GraphRuntimeOptions, str, str], GraphContext]
    build_views: Callable[[GraphInput], GraphViews]
    build_slices: Callable[[GraphViews, GraphContext], TSlices]
    build_rows: Callable[[str, str, GraphContext, GraphViews, TSlices], list[TRow]]


@dataclass(frozen=True)
class MetricsPipelineRequest:
    """Request parameters for graph metrics pipelines.

    Attributes
    ----------
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    graph
        Source graph to analyze.
    runtime
        Optional runtime options including backend selection.
    filters
        Filters for restricting graph nodes.
    """

    repo: str
    commit: str
    graph: GraphInput
    runtime: GraphRuntimeOptions | None = None
    filters: GraphFilterProtocol | None = None


@dataclass(frozen=True)
class _NoOpGraphFilters:
    @staticmethod
    def filter_call_graph(graph: GraphInput) -> GraphInput:
        return graph

    @staticmethod
    def filter_import_graph(graph: GraphInput) -> GraphInput:
        return graph

    @staticmethod
    def filter_subsystem_graph(graph: GraphInput) -> GraphInput:
        return graph

    @staticmethod
    def filter_subsystem_memberships(
        memberships: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        return memberships


_NO_OP_GRAPH_FILTERS: GraphFilterProtocol = _NoOpGraphFilters()


def _copy_without_self_loops(store: RxGraphStore) -> RxGraphStore:
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
    for node_id in store.node_ids():
        filtered.set_node_attrs(node_id, store.get_node_attrs(node_id))
    for src_idx, dst_idx in store.graph.edge_list():
        if src_idx == dst_idx:
            continue
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        filtered.add_weighted_edge(src_id, dst_id, weight=weight)
    return filtered


def build_graph_views(source_graph: GraphInput) -> GraphViews:
    """Build standard graph views from a source directed graph.

    Create the three graph representations needed for extended metrics:
    the original graph, a simplified version without self-loops, and
    an undirected view for structural analysis.

    Parameters
    ----------
    source_graph
        The source directed graph (call graph or import graph).

    Returns
    -------
    GraphViews
        Dataclass containing graph, simple_graph, and undirected views.
    """
    graph_store = ensure_directed_store(source_graph)
    simple_graph = _copy_without_self_loops(graph_store)
    undirected = to_undirected_store(simple_graph)
    return GraphViews(graph=graph_store, simple_graph=simple_graph, undirected=undirected)


def build_store_views(source_graph: GraphInput) -> GraphViews:
    """Build standard graph views from a graph input.

    Parameters
    ----------
    source_graph
        Graph input (directed or undirected).

    Returns
    -------
    GraphViews
        Graph, simplified graph, and undirected views.
    """
    graph_store = ensure_store(source_graph)
    simple_graph = _copy_without_self_loops(graph_store)
    undirected = to_undirected_store(simple_graph)
    return GraphViews(graph=graph_store, simple_graph=simple_graph, undirected=undirected)


def build_extended_metrics_rows[TSlices, TRow: Mapping[str, object]](
    config: ExtendedMetricsConfig[TSlices, TRow],
    request: ExtendedMetricsRequest,
) -> list[TRow]:
    """Execute the generic extended graph metrics computation workflow.

    Implement the common orchestration pattern for computing extended
    graph metrics. Handle runtime option resolution, graph filtering,
    view construction, and delegate to config-specific callables for slice
    computation and row building.

    Parameters
    ----------
    config
        Configuration specifying domain-specific callables for this metric type.
    request
        Request parameters including repo, commit, graph, runtime, and filters.

    Returns
    -------
    list[TRow]
        Rows produced by the extended metrics pipeline.
    """
    runtime_opts = request.runtime or GraphRuntimeOptions()
    active_filters = request.filters or _NO_OP_GRAPH_FILTERS
    ctx = config.build_context(runtime_opts, request.repo, request.commit)
    filtered_graph = config.filter_graph(active_filters, request.graph)
    views = build_graph_views(filtered_graph)
    slices = config.build_slices(views, ctx)
    return config.build_rows(request.repo, request.commit, ctx, views, slices)


def build_metrics_pipeline_rows[TSlices, TRow](
    config: MetricsPipelineConfig[TSlices, TRow],
    request: MetricsPipelineRequest,
) -> list[TRow]:
    """Execute a metrics pipeline for graph analytics.

    Parameters
    ----------
    config
        Configuration specifying the pipeline stages.
    request
        Request parameters including repo, commit, graph, runtime, and filters.

    Returns
    -------
    list[TRow]
        Rows produced by the metrics pipeline.
    """
    runtime_opts = request.runtime or GraphRuntimeOptions()
    active_filters = request.filters or _NO_OP_GRAPH_FILTERS
    ctx = config.build_context(runtime_opts, request.repo, request.commit)
    filtered_graph = config.filter_graph(active_filters, request.graph)
    views = config.build_views(filtered_graph)
    slices = config.build_slices(views, ctx)
    return config.build_rows(request.repo, request.commit, ctx, views, slices)


__all__ = [
    "ExtendedMetricsConfig",
    "ExtendedMetricsRequest",
    "GraphViews",
    "MetricsPipelineConfig",
    "MetricsPipelineRequest",
    "build_extended_metrics_rows",
    "build_graph_views",
    "build_metrics_pipeline_rows",
    "build_store_views",
]
