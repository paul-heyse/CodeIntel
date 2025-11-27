"""Subsystem-level graph metrics derived from the import graph."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import GraphContext, centrality_directed
from codeintel.graphs.engine import GraphEngine
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import ensure_schema


def _dag_layers(graph: nx.DiGraph) -> dict[str, int]:
    layers: dict[str, int] = {node: 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def _subsystem_centralities(
    graph: nx.DiGraph,
    ctx: GraphContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}, {}, {}
    centrality = centrality_directed(graph, ctx)
    return centrality.pagerank, centrality.betweenness, centrality.closeness


def _layer_by_subsystem(subsystem_graph: nx.DiGraph) -> dict[str, int]:
    condensation = nx.condensation(subsystem_graph)
    layers = _dag_layers(condensation)
    scc_index = condensation.graph.get("mapping", {})
    layer_map: dict[str, int] = {}
    for node in subsystem_graph.nodes:
        comp_idx = scc_index.get(node)
        layer_map[node] = layers.get(comp_idx, 0) if comp_idx is not None else 0
    return layer_map


def _degree_maps(
    subsystem_graph: nx.DiGraph, *, weight: str | None
) -> tuple[dict[str, float], dict[str, float]]:
    in_degree_pairs = cast("Iterable[tuple[str, float]]", subsystem_graph.in_degree(weight=weight))
    out_degree_pairs = cast(
        "Iterable[tuple[str, float]]", subsystem_graph.out_degree(weight=weight)
    )
    return (
        {str(node): float(deg) for node, deg in in_degree_pairs},
        {str(node): float(deg) for node, deg in out_degree_pairs},
    )


def compute_subsystem_graph_metrics(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    runtime: GraphRuntimeOptions | None = None,
) -> None:
    """Build subsystem-level condensed import graph metrics."""
    runtime = runtime or GraphRuntimeOptions()
    graph_ctx = runtime.graph_ctx
    con = gateway.con
    ensure_schema(con, "analytics.subsystem_graph_metrics")
    graph_ctx = graph_ctx or GraphContext(
        repo=repo,
        commit=commit,
        now=datetime.now(UTC),
        use_gpu=runtime.use_gpu,
    )
    if graph_ctx.use_gpu != runtime.use_gpu:
        graph_ctx = replace(graph_ctx, use_gpu=runtime.use_gpu)

    if runtime.context is not None and (
        runtime.context.repo != repo or runtime.context.commit != commit
    ):
        return

    membership_rows = con.execute(
        """
        SELECT subsystem_id, module
        FROM analytics.subsystem_modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    if not membership_rows:
        con.execute(
            "DELETE FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        return

    module_to_subsystem: dict[str, str] = {
        str(module): str(subsystem_id) for subsystem_id, module in membership_rows
    }
    subsystem_graph = nx.DiGraph()
    subsystem_graph.add_nodes_from({subsystem_id for subsystem_id, _ in membership_rows})

    engine: GraphEngine = runtime.build_engine(gateway, repo, commit)
    for src, dst, data in engine.import_graph().edges(data=True):
        src_sub = module_to_subsystem.get(src)
        dst_sub = module_to_subsystem.get(dst)
        if src_sub is None or dst_sub is None or src_sub == dst_sub:
            continue
        weight = float(data.get(graph_ctx.betweenness_weight or "weight", 1.0))
        if subsystem_graph.has_edge(src_sub, dst_sub):
            subsystem_graph[src_sub][dst_sub]["weight"] += weight
        else:
            subsystem_graph.add_edge(src_sub, dst_sub, weight=weight)

    if subsystem_graph.number_of_nodes() == 0:
        con.execute(
            "DELETE FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        return

    centralities = _subsystem_centralities(subsystem_graph, graph_ctx)
    layer_by_subsystem = _layer_by_subsystem(subsystem_graph)
    degree_maps = _degree_maps(subsystem_graph, weight=graph_ctx.betweenness_weight)

    now = graph_ctx.resolved_now()
    rows = [
        (
            repo,
            commit,
            subsystem,
            float(degree_maps[0].get(subsystem, 0.0)),
            float(degree_maps[1].get(subsystem, 0.0)),
            centralities[0].get(subsystem, 0.0),
            centralities[1].get(subsystem, 0.0),
            centralities[2].get(subsystem, 0.0),
            layer_by_subsystem.get(subsystem, 0),
            now,
        )
        for subsystem in subsystem_graph.nodes
    ]

    con.execute(
        "DELETE FROM analytics.subsystem_graph_metrics WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.executemany(
        """
        INSERT INTO analytics.subsystem_graph_metrics (
            repo, commit, subsystem_id, import_in_degree, import_out_degree,
            import_pagerank, import_betweenness, import_closeness, import_layer, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
