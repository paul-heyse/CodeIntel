"""Extended NetworkX-derived metrics for the call graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    CentralityBundle,
    ComponentBundle,
    GraphContext,
    StructuralMetrics,
    centrality_directed,
    component_metadata,
    structural_metrics,
    to_decimal_id,
)
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

CENTRALITY_SAMPLE_LIMIT = 500
EIGEN_MAX_ITER = 200


@dataclass(frozen=True)
class GraphViews:
    """Graph variants used for function-level metrics."""

    graph: nx.DiGraph
    simple_graph: nx.DiGraph
    undirected: nx.Graph


@dataclass(frozen=True)
class FunctionGraphSlices:
    """Precomputed graph statistics shared across metric rows."""

    centralities: CentralityBundle
    structure: StructuralMetrics
    components: ComponentBundle
    articulations: set[int]
    bridge_incident: dict[int, int]


def _bridge_endpoint_counts(graph: nx.Graph) -> dict[int, int]:
    counts: dict[int, int] = dict.fromkeys(graph.nodes, 0)
    for left, right in nx.bridges(graph):
        counts[left] += 1
        counts[right] += 1
    return counts


def _resolve_function_context(runtime: GraphRuntimeOptions, repo: str, commit: str) -> GraphContext:
    ctx = runtime.graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        betweenness_sample=CENTRALITY_SAMPLE_LIMIT,
        eigen_max_iter=EIGEN_MAX_ITER,
        pagerank_weight="weight",
        betweenness_weight="weight",
        use_gpu=runtime.use_gpu,
    )
    if ctx.betweenness_sample > CENTRALITY_SAMPLE_LIMIT or ctx.eigen_max_iter > EIGEN_MAX_ITER:
        ctx = replace(
            ctx,
            betweenness_sample=min(ctx.betweenness_sample, CENTRALITY_SAMPLE_LIMIT),
            eigen_max_iter=min(ctx.eigen_max_iter, EIGEN_MAX_ITER),
        )
    if ctx.use_gpu != runtime.use_gpu:
        ctx = replace(ctx, use_gpu=runtime.use_gpu)
    return ctx


def _build_function_views(
    engine: GraphEngine,
) -> GraphViews:
    graph: nx.DiGraph = engine.call_graph()
    simple_graph: nx.DiGraph = cast("nx.DiGraph", graph.copy())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()
    return GraphViews(graph=graph, simple_graph=simple_graph, undirected=undirected)


def _function_metric_slices(views: GraphViews, ctx: GraphContext) -> FunctionGraphSlices:
    centralities = centrality_directed(views.simple_graph, ctx, include_eigen=True)
    structure = structural_metrics(views.undirected, weight=ctx.pagerank_weight)
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
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    node_count = views.graph.number_of_nodes()
    created_at = ctx.resolved_now()
    for node in views.simple_graph.nodes:
        goid = to_decimal_id(node)
        ancestor_count = len(nx.ancestors(views.graph, node)) if node_count else 0
        descendant_count = len(nx.descendants(views.graph, node)) if node_count else 0
        rows.append(
            (
                repo,
                commit,
                goid,
                slices.centralities.betweenness.get(node, 0.0),
                slices.centralities.closeness.get(node, 0.0),
                slices.centralities.eigenvector.get(node, 0.0),
                slices.centralities.harmonic.get(node, 0.0),
                slices.structure.core_number.get(node),
                slices.structure.clustering.get(node, 0.0),
                int(slices.structure.triangles.get(node, 0)),
                node in slices.articulations,
                None,
                slices.bridge_incident.get(node, 0) > 0,
                slices.components.component_id.get(node),
                slices.components.component_size.get(node),
                slices.components.scc_id.get(node),
                slices.components.scc_size.get(node),
                ancestor_count,
                descendant_count,
                slices.structure.community_id.get(node),
                created_at,
            )
        )
    return rows


def compute_graph_metrics_functions_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    """
    Populate analytics.graph_metrics_functions_ext with additional centralities.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection used for reads and writes.
    repo : str
        Repository identifier anchoring the metrics.
    commit : str
        Commit hash anchoring the metrics snapshot.
    runtime : GraphRuntimeOptions | None
        Optional runtime options including cached graphs and backend selection.
    """
    runtime = runtime or GraphRuntimeOptions()
    ctx = _resolve_function_context(runtime, repo, commit)
    engine = runtime.build_engine(gateway, repo, commit)
    views = _build_function_views(engine)
    slices = _function_metric_slices(views, ctx)
    rows = _function_metric_rows(repo, commit, ctx, views, slices)

    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions_ext")
    con.execute(
        "DELETE FROM analytics.graph_metrics_functions_ext WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.graph_metrics_functions_ext (
                repo, commit, function_goid_h128,
                call_betweenness, call_closeness, call_eigenvector, call_harmonic,
                call_core_number, call_clustering_coeff, call_triangle_count,
                call_is_articulation, call_articulation_impact, call_is_bridge_endpoint,
                call_component_id, call_component_size, call_scc_id, call_scc_size,
                call_ancestor_count, call_descendant_count, call_community_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
