"""Extended NetworkX-derived metrics for the call graph."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    GraphContext,
    centrality_directed,
    component_metadata,
    structural_metrics,
    to_decimal_id,
)
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_call_graph
from codeintel.storage.gateway import StorageGateway

CENTRALITY_SAMPLE_LIMIT = 500
EIGEN_MAX_ITER = 200


def _bridge_endpoint_counts(graph: nx.Graph) -> dict[int, int]:
    counts: dict[int, int] = dict.fromkeys(graph.nodes, 0)
    for left, right in nx.bridges(graph):
        counts[left] += 1
        counts[right] += 1
    return counts


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
    context = runtime.context
    graph_ctx = runtime.graph_ctx
    use_gpu = runtime.use_gpu
    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions_ext")
    ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        betweenness_sample=CENTRALITY_SAMPLE_LIMIT,
        eigen_max_iter=EIGEN_MAX_ITER,
        pagerank_weight="weight",
        betweenness_weight="weight",
        use_gpu=use_gpu,
    )
    if ctx.betweenness_sample > CENTRALITY_SAMPLE_LIMIT or ctx.eigen_max_iter > EIGEN_MAX_ITER:
        ctx = replace(
            ctx,
            betweenness_sample=min(ctx.betweenness_sample, CENTRALITY_SAMPLE_LIMIT),
            eigen_max_iter=min(ctx.eigen_max_iter, EIGEN_MAX_ITER),
        )
    if ctx.use_gpu != use_gpu:
        ctx = replace(ctx, use_gpu=use_gpu)
    graph: nx.DiGraph = (
        context.call_graph
        if context is not None
        else load_call_graph(gateway, repo, commit, use_gpu=use_gpu)
    )
    simple_graph: nx.DiGraph = cast("nx.DiGraph", graph.copy())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()

    centralities = centrality_directed(simple_graph, ctx, include_eigen=True)
    structure = structural_metrics(undirected, weight=ctx.pagerank_weight)
    components = component_metadata(simple_graph)
    articulations = (
        set(nx.articulation_points(undirected)) if undirected.number_of_nodes() > 0 else set()
    )
    bridge_incident = _bridge_endpoint_counts(undirected)

    rows: list[tuple[object, ...]] = []
    for node in simple_graph.nodes:
        goid = to_decimal_id(node)
        ancestor_count = len(nx.ancestors(graph, node)) if graph.number_of_nodes() else 0
        descendant_count = len(nx.descendants(graph, node)) if graph.number_of_nodes() else 0
        rows.append(
            (
                repo,
                commit,
                goid,
                centralities.betweenness.get(node, 0.0),
                centralities.closeness.get(node, 0.0),
                centralities.eigenvector.get(node, 0.0),
                centralities.harmonic.get(node, 0.0),
                structure.core_number.get(node),
                structure.clustering.get(node, 0.0),
                int(structure.triangles.get(node, 0)),
                node in articulations,
                None,  # placeholder for articulation impact
                bridge_incident.get(node, 0) > 0,
                components.component_id.get(node),
                components.component_size.get(node),
                components.scc_id.get(node),
                components.scc_size.get(node),
                ancestor_count,
                descendant_count,
                structure.community_id.get(node),
                ctx.resolved_now(),
            )
        )

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
