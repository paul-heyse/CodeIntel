"""Generic orchestration framework for extended graph metrics computation.

This module provides a reusable orchestration pattern for computing extended
graph metrics. It extracts the common workflow shared between function-level
and module-level extended metrics into a configurable generic function.

The orchestrator handles:
1. Runtime resolution and option extraction
2. Snapshot and filter construction
3. Graph retrieval and filtering
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
    get_source_graph=lambda rt: rt.ensure_call_graph(),
    filter_graph=lambda f, g: f.filter_call_graph(g),
    build_context=_resolve_function_context,
    build_slices=_function_metric_slices,
    build_rows=_function_metric_rows,
)

request = ExtendedMetricsRequest(repo="org/repo", commit="abc123")
build_extended_metrics_rows(gateway, config, request)
```
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import networkx as nx

from codeintel.build.analytics.graphs.graph_metrics import build_graph_metric_filters
from codeintel.build.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.build.graphs.runtime.context import GraphContext
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class GraphViews:
    """Graph variants used for graph metrics computation.

    This dataclass holds the three standard graph representations needed
    for computing extended graph metrics: the original directed graph,
    a simplified version without self-loops, and an undirected view.

    Attributes
    ----------
    graph
        The original directed graph (call graph or import graph).
    simple_graph
        Directed graph with self-loops removed.
    undirected
        Undirected view of the simple graph for structural metrics.
    """

    graph: nx.DiGraph
    simple_graph: nx.DiGraph
    undirected: nx.Graph


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
    get_source_graph
        Callable to retrieve the source graph from the resolved runtime.
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
    get_source_graph: Callable[[GraphRuntime], nx.DiGraph]
    filter_graph: Callable[[GraphMetricFilters, nx.DiGraph], nx.DiGraph]
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
    runtime
        Optional runtime options including cached graphs and backend selection.
    filters
        Optional allowlists for restricting graph nodes.
    """

    repo: str
    commit: str
    runtime: GraphRuntime | GraphRuntimeOptions | None = None
    filters: GraphMetricFilters | None = None


def build_graph_views(source_graph: nx.DiGraph) -> GraphViews:
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
    simple_graph = cast("nx.DiGraph", source_graph.copy())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()
    return GraphViews(graph=source_graph, simple_graph=simple_graph, undirected=undirected)


def build_extended_metrics_rows[TSlices, TRow: Mapping[str, object]](
    gateway: StorageGateway,
    config: ExtendedMetricsConfig[TSlices, TRow],
    request: ExtendedMetricsRequest,
) -> list[TRow]:
    """Execute the generic extended graph metrics computation workflow.

    Implement the common orchestration pattern for computing extended
    graph metrics. Handle runtime resolution, graph retrieval, filtering,
    view construction, and delegate to config-specific callables for slice
    computation and row building.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection used for reads and writes.
    config
        Configuration specifying domain-specific callables for this metric type.
    request
        Request parameters including repo, commit, runtime, and filters.

    Returns
    -------
    list[TRow]
        Rows produced by the extended metrics pipeline.
    """
    # 1. Resolve runtime options
    runtime_opts = (
        request.runtime.options
        if isinstance(request.runtime, GraphRuntime)
        else request.runtime or GraphRuntimeOptions()
    )
    # 2. Build snapshot
    snapshot = runtime_opts.snapshot or SnapshotRef(
        repo=request.repo, commit=request.commit, repo_root=Path()
    )
    # 3. Build filters
    active_filters = request.filters or build_graph_metric_filters(gateway, snapshot)
    # 4. Resolve graph runtime
    resolved_runtime = resolve_graph_runtime(gateway, snapshot, runtime_opts)
    # 5. Build context
    ctx = config.build_context(runtime_opts, request.repo, request.commit)
    # 6. Get and filter graph
    source_graph = config.get_source_graph(resolved_runtime)
    filtered_graph = config.filter_graph(active_filters, source_graph)
    # 7. Build views
    views = build_graph_views(filtered_graph)
    # 8. Compute slices
    slices = config.build_slices(views, ctx)
    # 9. Build rows
    return config.build_rows(request.repo, request.commit, ctx, views, slices)


__all__ = [
    "ExtendedMetricsConfig",
    "ExtendedMetricsRequest",
    "GraphViews",
    "build_extended_metrics_rows",
    "build_graph_views",
]
