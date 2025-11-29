"""Extended module-level import graph metrics using NetworkX."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import networkx as nx

from codeintel.analytics.graph_rows import (
    ModuleMetricExtInputs,
    ModuleMetricExtRow,
    build_module_metric_ext_rows,
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
from codeintel.analytics.graphs.graph_metrics import GraphMetricFilters, build_graph_metric_filters
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig
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
    return resolve_graph_context(
        GraphContextSpec(
            repo=repo,
            commit=commit,
            use_gpu=runtime.use_gpu,
            now=datetime.now(UTC),
            betweenness_cap=CENTRALITY_SAMPLE_LIMIT,
            pagerank_weight="weight",
            betweenness_weight="weight",
            community_detection_limit=runtime.features.community_detection_limit,
        )
    )


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
    structure = structural_metrics(
        views.undirected,
        weight=ctx.pagerank_weight,
        community_limit=ctx.community_detection_limit,
    )
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
) -> list[ModuleMetricExtRow]:
    centralities = {
        "betweenness": slices.centralities.betweenness,
        "closeness": slices.centralities.closeness,
        "eigenvector": slices.centralities.eigenvector,
        "harmonic": slices.centralities.harmonic,
    }
    structure = {
        "core_number": slices.structure.core_number,
        "constraint": slices.structure.constraint,
        "effective_size": slices.structure.effective_size,
        "community_id": slices.structure.community_id,
    }
    components = {
        "component_id": slices.components.component_id,
        "component_size": slices.components.component_size,
        "scc_id": slices.components.scc_id,
        "scc_size": slices.components.scc_size,
    }
    rich_club = {
        module: slices.degree_map.get(module, 0) >= slices.degree_cutoff
        if slices.degree_cutoff > 0
        else False
        for module in views.simple_graph.nodes
    }
    inputs = ModuleMetricExtInputs(
        repo=repo,
        commit=commit,
        ctx=ctx,
        centralities=centralities,
        structure=structure,
        components=components,
        rich_club=rich_club,
        nodes=sorted(views.simple_graph.nodes),
    )
    return build_module_metric_ext_rows(inputs)


def compute_graph_metrics_modules_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
    filters: GraphMetricFilters | None = None,
) -> None:
    """Populate analytics.graph_metrics_modules_ext with richer import metrics."""
    runtime_opts = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime or GraphRuntimeOptions()
    )
    snapshot = runtime_opts.snapshot or SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    cfg = GraphMetricsStepConfig(snapshot=snapshot)
    active_filters = filters or build_graph_metric_filters(gateway, cfg)
    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_opts,
        context=runtime_opts.context,
    )
    ctx = _resolve_module_context(runtime_opts, repo, commit)
    filtered_graph: nx.DiGraph = active_filters.filter_import_graph(
        resolved_runtime.ensure_import_graph()
    )
    simple_graph: nx.DiGraph = cast("nx.DiGraph", filtered_graph.copy())
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()
    views = ImportGraphViews(graph=filtered_graph, simple_graph=simple_graph, undirected=undirected)
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
