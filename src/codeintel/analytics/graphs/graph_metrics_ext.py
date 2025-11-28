"""Extended NetworkX-derived metrics for the call graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import networkx as nx

from codeintel.analytics.graph_rows import (
    FunctionMetricExtInputs,
    FunctionMetricExtRow,
    build_function_metric_ext_rows,
)
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graph_service import (
    CentralityBundle,
    ComponentBundle,
    GraphContext,
    StructuralMetrics,
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.analytics.graph_service_runtime import GraphContextSpec, resolve_graph_context
from codeintel.config.primitives import SnapshotRef
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
        )
    )


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
) -> list[FunctionMetricExtRow]:
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


def compute_graph_metrics_functions_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
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
    runtime : GraphRuntime | GraphRuntimeOptions | None
        Optional runtime options including cached graphs and backend selection.
    """
    runtime_opts = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    snapshot = runtime_opts.snapshot or SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_opts,
        context=runtime_opts.context,
    )
    ctx = _resolve_function_context(runtime_opts, repo, commit)
    engine = resolved_runtime.engine
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
