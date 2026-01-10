"""Extended call graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.graphs import (
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.build.analytics.compute.row_builders import (
    FunctionMetricExtInputs,
    RowBuildContext,
    build_function_metric_ext_rows,
)
from codeintel.build.analytics.graphs.constants import (
    CENTRALITY_SAMPLE_LIMIT,
    EIGEN_MAX_ITER,
)
from codeintel.build.analytics.graphs.context_helpers import GraphContextFactory
from codeintel.build.analytics.graphs.orchestrator import (
    ExtendedMetricsConfig,
    ExtendedMetricsRequest,
    build_extended_metrics_rows,
)
from codeintel.build.graphs.compute.metrics.components import (
    find_articulation_points,
    find_bridges,
)
from codeintel.build.graphs.runtime.context import GraphContext
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    ancestors_by_id,
    descendants_by_id,
    ensure_directed_store,
    graph_node_count,
    graph_to_store,
)
from codeintel.core.columnar.rows import ColumnarRowBuffer

if TYPE_CHECKING:
    from codeintel.build.analytics.compute.graphs import (
        CentralityBundle,
        ComponentBundle,
        StructuralMetrics,
    )
    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.build.analytics.graphs.orchestrator import (
        GraphViews,
    )
    from codeintel.build.graphs.runtime import GraphRuntimeOptions


@dataclass(frozen=True)
class FunctionGraphSlices:
    """Precomputed graph statistics shared across metric rows."""

    centralities: CentralityBundle
    structure: StructuralMetrics
    components: ComponentBundle
    articulations: set[int]
    bridge_incident: dict[int, int]


def _bridge_endpoint_counts(graph: GraphInput) -> dict[int, int]:
    """Count bridge endpoints for each node.

    Parameters
    ----------
    graph
        Undirected graph to analyze for bridges.

    Returns
    -------
    dict[int, int]
        Mapping of node to count of incident bridges.
    """
    store = graph_to_store(graph)
    counts: dict[int, int] = {int(str(node)): 0 for node in store.node_ids()}
    for left, right in find_bridges(store):
        left_idx = int(str(left))
        right_idx = int(str(right))
        counts[left_idx] = counts.get(left_idx, 0) + 1
        counts[right_idx] = counts.get(right_idx, 0) + 1
    return counts


def _resolve_function_context(runtime: GraphRuntimeOptions, repo: str, commit: str) -> GraphContext:
    """Build graph context with function-specific constants.

    Returns
    -------
    GraphContext
        Graph context configured for function-level metrics.
    """
    return _FUNCTION_CONTEXT_FACTORY.build(runtime, repo=repo, commit=commit)


def _function_metric_slices(views: GraphViews, ctx: GraphContext) -> FunctionGraphSlices:
    """Compute metric slices for function-level extended metrics.

    Parameters
    ----------
    views
        Graph views containing directed, simplified, and undirected graphs.
    ctx
        Graph context with computation parameters.

    Returns
    -------
    FunctionGraphSlices
        Precomputed statistics for row building.
    """
    centralities = centrality_directed(views.simple_graph, ctx, include_eigen=True)
    structure = structural_metrics(
        views.undirected,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
    components = component_metadata(views.simple_graph)
    articulations = {int(str(node)) for node in find_articulation_points(views.undirected)}
    bridge_incident = _bridge_endpoint_counts(views.undirected)
    return FunctionGraphSlices(
        centralities=centralities,
        structure=structure,
        components=components,
        articulations=articulations,
        bridge_incident=bridge_incident,
    )


def _function_metric_rows(
    repo: str,
    commit: str,
    ctx: GraphContext,
    views: GraphViews,
    slices: FunctionGraphSlices,
) -> ColumnarRowBuffer:
    """Build rows for function-level extended metrics.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    ctx
        Graph context with computation parameters.
    views
        Graph views for ancestor/descendant computation.
    slices
        Precomputed metric slices.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows ready for insertion.
    """
    graph_store = ensure_directed_store(views.graph)
    simple_store = ensure_directed_store(views.simple_graph)
    node_count = graph_node_count(graph_store)
    ancestor_count: dict[int, int] = {}
    descendant_count: dict[int, int] = {}
    for node_id in simple_store.node_ids():
        node_idx = graph_store.id_to_index.get(node_id)
        if node_idx is None:
            continue
        node_key = int(str(node_id))
        if node_count == 0:
            ancestor_count[node_key] = 0
            descendant_count[node_key] = 0
            continue
        ancestors = ancestors_by_id(graph_store, node_id)
        descendants = descendants_by_id(graph_store, node_id)
        ancestor_count[node_key] = len(ancestors)
        descendant_count[node_key] = len(descendants)
    centralities = {
        "betweenness": slices.centralities.betweenness,
        "closeness": slices.centralities.closeness,
        "eigenvector": slices.centralities.eigenvector,
        "harmonic": slices.centralities.harmonic,
    }
    structure = {
        "core_number": slices.structure.core_number,
        "clustering": slices.structure.clustering,
        "triangles": slices.structure.triangles,
        "community_id": slices.structure.community_id,
    }
    components = {
        "component_id": slices.components.component_id,
        "component_size": slices.components.component_size,
        "scc_id": slices.components.scc_id,
        "scc_size": slices.components.scc_size,
    }
    row_context = RowBuildContext.from_repo_commit(
        repo,
        commit,
        created_at=ctx.resolved_now(),
    )
    inputs = FunctionMetricExtInputs(
        row_context=row_context,
        ctx=ctx,
        centralities=centralities,
        structure=structure,
        components=components,
        articulations=slices.articulations,
        bridge_incident=slices.bridge_incident,
        ancestor_count=ancestor_count,
        descendant_count=descendant_count,
    )
    return build_function_metric_ext_rows(inputs)


# Context factory for function-level extended metrics
_FUNCTION_CONTEXT_FACTORY = GraphContextFactory(
    betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
    eigen_cap=EIGEN_MAX_ITER,
    pagerank_weight="weight",
    betweenness_weight="weight",
)

# Configuration for function-level extended metrics
_FUNCTION_EXT_CONFIG: ExtendedMetricsConfig[FunctionGraphSlices, ColumnarRowBuffer] = (
    ExtendedMetricsConfig(
        table_key="analytics.graph_metrics_functions_ext",
        filter_graph=lambda f, g: f.filter_call_graph(g),
        build_context=_resolve_function_context,
        build_slices=_function_metric_slices,
        build_rows=_function_metric_rows,
    )
)


def build_graph_metrics_functions_ext_rows(
    *,
    repo: str,
    commit: str,
    call_graph: GraphInput,
    runtime: GraphRuntimeOptions | None = None,
    filters: GraphMetricFilters | None = None,
) -> ColumnarRowBuffer:
    """Populate analytics.graph_metrics_functions_ext with additional centralities.

    Parameters
    ----------
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    call_graph
        Call graph to analyze for extended metrics.
    runtime
        Optional runtime options including cached graphs and backend selection.
    filters
        Optional allowlists for restricting graph nodes.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing rows for analytics.graph_metrics_functions_ext.
    """
    request = ExtendedMetricsRequest(
        repo=repo,
        commit=commit,
        graph=call_graph,
        runtime=runtime,
        filters=filters,
    )
    return build_extended_metrics_rows(_FUNCTION_EXT_CONFIG, request)
