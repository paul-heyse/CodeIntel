"""Subsystem-level graph metrics derived from the import graph."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import cast

import networkx as nx

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_import_graph
from codeintel.storage.gateway import StorageGateway


def _dag_layers(graph: nx.DiGraph) -> dict[str, int]:
    layers: dict[str, int] = {node: 0 for node in graph.nodes if graph.in_degree(node) == 0}
    for node in nx.topological_sort(graph):
        base = layers.get(node, 0)
        for succ in graph.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return layers


def _subsystem_centralities(
    graph: nx.DiGraph,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}, {}, {}
    pagerank = nx.pagerank(graph, weight="weight")
    betweenness = nx.betweenness_centrality(graph, weight="weight")
    closeness = nx.closeness_centrality(graph)
    return pagerank, betweenness, closeness


def _layer_by_subsystem(subsystem_graph: nx.DiGraph) -> dict[str, int]:
    condensation = nx.condensation(subsystem_graph)
    layers = _dag_layers(condensation)
    scc_index = condensation.graph.get("mapping", {})
    layer_map: dict[str, int] = {}
    for node in subsystem_graph.nodes:
        comp_idx = scc_index.get(node)
        layer_map[node] = layers.get(comp_idx, 0) if comp_idx is not None else 0
    return layer_map


def _degree_maps(subsystem_graph: nx.DiGraph) -> tuple[dict[str, float], dict[str, float]]:
    in_degree_pairs = cast(
        "Iterable[tuple[str, float]]", subsystem_graph.in_degree(weight="weight")
    )
    out_degree_pairs = cast(
        "Iterable[tuple[str, float]]", subsystem_graph.out_degree(weight="weight")
    )
    return (
        {str(node): float(deg) for node, deg in in_degree_pairs},
        {str(node): float(deg) for node, deg in out_degree_pairs},
    )


def compute_subsystem_graph_metrics(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Build subsystem-level condensed import graph metrics."""
    con = gateway.con
    ensure_schema(con, "analytics.subsystem_graph_metrics")

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

    for src, dst, data in load_import_graph(gateway, repo, commit).edges(data=True):
        src_sub = module_to_subsystem.get(src)
        dst_sub = module_to_subsystem.get(dst)
        if src_sub is None or dst_sub is None or src_sub == dst_sub:
            continue
        weight = float(data.get("weight", 1.0))
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

    centralities = _subsystem_centralities(subsystem_graph)
    layer_by_subsystem = _layer_by_subsystem(subsystem_graph)
    in_degree, out_degree = _degree_maps(subsystem_graph)

    now = datetime.now(UTC)
    rows = [
        (
            repo,
            commit,
            subsystem,
            float(in_degree.get(subsystem, 0.0)),
            float(out_degree.get(subsystem, 0.0)),
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
