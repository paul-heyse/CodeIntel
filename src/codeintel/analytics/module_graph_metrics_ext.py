"""Extended module-level import graph metrics using NetworkX."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_service import (
    GraphBundle,
    GraphContext,
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_import_graph
from codeintel.storage.gateway import StorageGateway

CENTRALITY_SAMPLE_LIMIT = 500
RICH_CLUB_PERCENTILE = 0.1


def _rich_club_cutoff(degree_map: dict[object, int]) -> int:
    if not degree_map:
        return 0
    sorted_degrees = sorted(degree_map.values(), reverse=True)
    idx = max(0, int(len(sorted_degrees) * RICH_CLUB_PERCENTILE) - 1)
    return sorted_degrees[idx] if idx < len(sorted_degrees) else sorted_degrees[-1]


def compute_graph_metrics_modules_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    context: AnalyticsContext | None = None,
    graph_ctx: GraphContext | None = None,
) -> None:
    """Populate analytics.graph_metrics_modules_ext with richer import metrics."""
    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_modules_ext")

    ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        betweenness_sample=CENTRALITY_SAMPLE_LIMIT,
        pagerank_weight="weight",
        betweenness_weight="weight",
    )
    if ctx.betweenness_sample > CENTRALITY_SAMPLE_LIMIT:
        ctx = replace(ctx, betweenness_sample=CENTRALITY_SAMPLE_LIMIT)
    import_graph_cached = context.import_graph if context is not None else None

    def _import_graph_loader() -> nx.DiGraph:
        return (
            import_graph_cached
            if import_graph_cached is not None
            else load_import_graph(gateway, repo, commit)
        )

    bundle: GraphBundle[nx.DiGraph] = GraphBundle(
        ctx=ctx,
        loaders={"import_graph": _import_graph_loader},
    )
    graph: nx.DiGraph = bundle.get("import_graph")
    simple_graph: nx.DiGraph = cast("nx.DiGraph", graph.copy())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()

    centralities = centrality_directed(simple_graph, ctx, include_eigen=True)
    structure = structural_metrics(undirected, weight=ctx.pagerank_weight)
    components = component_metadata(simple_graph)

    degree_view = cast("nx.classes.reportviews.DegreeView", simple_graph.degree)
    degree_map = {node: int(degree_view[node]) for node in simple_graph.nodes}
    degree_cutoff = _rich_club_cutoff(degree_map)

    rows: list[tuple[object, ...]] = [
        (
            repo,
            commit,
            module,
            centralities.betweenness.get(module, 0.0),
            centralities.closeness.get(module, 0.0),
            centralities.eigenvector.get(module, 0.0),
            centralities.harmonic.get(module, 0.0),
            structure.core_number.get(module),
            structure.constraint.get(module),
            structure.effective_size.get(module),
            degree_map.get(module, 0) >= degree_cutoff if degree_cutoff > 0 else False,
            structure.core_number.get(module),
            structure.community_id.get(module),
            components.component_id.get(module),
            components.component_size.get(module),
            components.scc_id.get(module),
            components.scc_size.get(module),
            ctx.resolved_now(),
        )
        for module in simple_graph.nodes
    ]

    con.execute(
        "DELETE FROM analytics.graph_metrics_modules_ext WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.graph_metrics_modules_ext (
                repo, commit, module,
                import_betweenness, import_closeness, import_eigenvector, import_harmonic,
                import_k_core, import_constraint, import_effective_size,
                import_rich_club, import_shell_index,
                import_community_id, import_component_id, import_component_size,
                import_scc_id, import_scc_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
