"""Extended module-level import graph metrics using NetworkX."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.graph_service import (
    CentralityBundle,
    ComponentBundle,
    GraphContext,
    StructuralMetrics,
    centrality_directed,
    component_metadata,
    structural_metrics,
)
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

CENTRALITY_SAMPLE_LIMIT = 500
RICH_CLUB_PERCENTILE = 0.1


@dataclass(frozen=True)
class ImportGraphViews:
    """Graph variants used for module-level metrics."""

    graph: nx.DiGraph
    simple_graph: nx.DiGraph
    undirected: nx.Graph


@dataclass(frozen=True)
class ModuleGraphSlices:
    """Precomputed graph statistics used for module metrics."""

    centralities: CentralityBundle
    structure: StructuralMetrics
    components: ComponentBundle
    degree_map: dict[object, int]
    degree_cutoff: int


def _rich_club_cutoff(degree_map: dict[object, int]) -> int:
    if not degree_map:
        return 0
    sorted_degrees = sorted(degree_map.values(), reverse=True)
    idx = max(0, int(len(sorted_degrees) * RICH_CLUB_PERCENTILE) - 1)
    return sorted_degrees[idx] if idx < len(sorted_degrees) else sorted_degrees[-1]


def _resolve_module_context(runtime: GraphRuntimeOptions, repo: str, commit: str) -> GraphContext:
    ctx = runtime.graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        betweenness_sample=CENTRALITY_SAMPLE_LIMIT,
        pagerank_weight="weight",
        betweenness_weight="weight",
        use_gpu=runtime.use_gpu,
    )
    if ctx.betweenness_sample > CENTRALITY_SAMPLE_LIMIT:
        ctx = replace(ctx, betweenness_sample=CENTRALITY_SAMPLE_LIMIT)
    if ctx.use_gpu != runtime.use_gpu:
        ctx = replace(ctx, use_gpu=runtime.use_gpu)
    return ctx


def _build_import_views(
    engine: GraphEngine,
) -> ImportGraphViews:
    graph: nx.DiGraph = engine.import_graph()
    simple_graph: nx.DiGraph = cast("nx.DiGraph", graph.copy())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()
    return ImportGraphViews(graph=graph, simple_graph=simple_graph, undirected=undirected)


def _module_metric_slices(views: ImportGraphViews, ctx: GraphContext) -> ModuleGraphSlices:
    centralities = centrality_directed(views.simple_graph, ctx, include_eigen=True)
    structure = structural_metrics(views.undirected, weight=ctx.pagerank_weight)
    components = component_metadata(views.simple_graph)
    degree_view = cast("Iterable[tuple[object, float]]", views.simple_graph.degree)
    degree_map: dict[object, int] = {node: int(deg) for node, deg in degree_view}
    return ModuleGraphSlices(
        centralities=centralities,
        structure=structure,
        components=components,
        degree_map=degree_map,
        degree_cutoff=_rich_club_cutoff(degree_map),
    )


def _module_metric_rows(
    repo: str,
    commit: str,
    ctx: GraphContext,
    views: ImportGraphViews,
    slices: ModuleGraphSlices,
) -> list[tuple[object, ...]]:
    created_at = ctx.resolved_now()
    return [
        (
            repo,
            commit,
            module,
            slices.centralities.betweenness.get(module, 0.0),
            slices.centralities.closeness.get(module, 0.0),
            slices.centralities.eigenvector.get(module, 0.0),
            slices.centralities.harmonic.get(module, 0.0),
            slices.structure.core_number.get(module),
            slices.structure.constraint.get(module),
            slices.structure.effective_size.get(module),
            slices.degree_map.get(module, 0) >= slices.degree_cutoff
            if slices.degree_cutoff > 0
            else False,
            slices.structure.core_number.get(module),
            slices.structure.community_id.get(module),
            slices.components.component_id.get(module),
            slices.components.component_size.get(module),
            slices.components.scc_id.get(module),
            slices.components.scc_size.get(module),
            created_at,
        )
        for module in views.simple_graph.nodes
    ]


def compute_graph_metrics_modules_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """Populate analytics.graph_metrics_modules_ext with richer import metrics."""
    runtime_opts = runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    ctx = _resolve_module_context(runtime_opts, repo, commit)
    if isinstance(runtime, GraphRuntime):
        engine = runtime.engine
    else:
        engine = runtime_opts.build_engine(gateway, repo, commit)
    views = _build_import_views(engine)
    slices = _module_metric_slices(views, ctx)
    rows = _module_metric_rows(repo, commit, ctx, views, slices)

    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_modules_ext")

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
