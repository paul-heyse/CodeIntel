"""Extended NetworkX-derived metrics for the call graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.analytics.compute.graphs import (
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.analytics.compute.row_builders import (
    FunctionMetricExtInputs,
    build_function_metric_ext_rows,
)
from codeintel.analytics.graphs.constants import (
    CENTRALITY_SAMPLE_LIMIT,
    EIGEN_MAX_ITER,
)
from codeintel.analytics.graphs.orchestrator import (
    ExtendedMetricsConfig,
    ExtendedMetricsRequest,
    build_extended_metrics_rows,
)
from codeintel.graphs.runtime.context import GraphContextSpec, resolve_graph_context

if TYPE_CHECKING:
    from codeintel.analytics.compute.graphs import (
        CentralityBundle,
        ComponentBundle,
        StructuralMetrics,
    )
    from codeintel.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.analytics.graphs.orchestrator import (
        GraphViews,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsGraphMetricsFunctionsExtRow as GraphMetricsFunctionsExtRow,
    )
    from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions
    from codeintel.graphs.runtime.context import GraphContext
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class FunctionGraphSlices:
    """Precomputed graph statistics shared across metric rows."""

    centralities: CentralityBundle
    structure: StructuralMetrics
    components: ComponentBundle
    articulations: set[int]
    bridge_incident: dict[int, int]


def _bridge_endpoint_counts(graph: nx.Graph) -> dict[int, int]:
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
    counts: dict[int, int] = dict.fromkeys(graph.nodes, 0)
    for left, right in nx.bridges(graph):
        counts[left] += 1
        counts[right] += 1
    return counts


def _resolve_function_context(runtime: GraphRuntimeOptions, repo: str, commit: str) -> GraphContext:
    """Build graph context with function-specific constants.

    Parameters
    ----------
    runtime
        Runtime options including GPU and community detection settings.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    GraphContext
        Resolved graph context for function metrics computation.
    """
    return resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=runtime.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
            eigen_cap=EIGEN_MAX_ITER,
            pagerank_weight="weight",
            betweenness_weight="weight",
            community_detection_limit=runtime.features.community_detection_limit,
        )
    )


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
    articulations = (
        set(nx.articulation_points(views.undirected))
        if views.undirected.number_of_nodes() > 0
        else set()
    )
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
) -> list[GraphMetricsFunctionsExtRow]:
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
    list[GraphMetricsFunctionsExtRow]
        Rows ready for insertion.
    """
    node_count = views.graph.number_of_nodes()
    ancestor_count = {
        node: len(nx.ancestors(views.graph, node)) if node_count else 0
        for node in views.simple_graph.nodes
    }
    descendant_count = {
        node: len(nx.descendants(views.graph, node)) if node_count else 0
        for node in views.simple_graph.nodes
    }
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
    inputs = FunctionMetricExtInputs(
        repo=repo,
        commit=commit,
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


# Configuration for function-level extended metrics
_FUNCTION_EXT_CONFIG: ExtendedMetricsConfig[FunctionGraphSlices, GraphMetricsFunctionsExtRow] = (
    ExtendedMetricsConfig(
        table_key="analytics.graph_metrics_functions_ext",
        get_source_graph=lambda rt: rt.ensure_call_graph(),
        filter_graph=lambda f, g: f.filter_call_graph(g),
        build_context=_resolve_function_context,
        build_slices=_function_metric_slices,
        build_rows=_function_metric_rows,
    )
)


def build_graph_metrics_functions_ext_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
    filters: GraphMetricFilters | None = None,
) -> list[GraphMetricsFunctionsExtRow]:
    """Populate analytics.graph_metrics_functions_ext with additional centralities.

    Parameters
    ----------
    gateway
        StorageGateway providing the DuckDB connection used for reads and writes.
    repo
        Repository identifier anchoring the metrics.
    commit
        Commit hash anchoring the metrics snapshot.
    runtime
        Optional runtime options including cached graphs and backend selection.
    filters
        Optional allowlists for restricting graph nodes.

    Returns
    -------
    list[GraphMetricsFunctionsExtRow]
        Rows ready for insertion into analytics.graph_metrics_functions_ext.
    """
    request = ExtendedMetricsRequest(
        repo=repo,
        commit=commit,
        runtime=runtime,
        filters=filters,
    )
    return build_extended_metrics_rows(gateway, _FUNCTION_EXT_CONFIG, request)
