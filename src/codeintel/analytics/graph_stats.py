"""Global graph statistics for core graphs."""

from __future__ import annotations

from datetime import UTC, datetime

import networkx as nx
from networkx.algorithms import approximation
from networkx.exception import NetworkXError

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import (
    load_call_graph,
    load_config_module_bipartite,
    load_import_graph,
    load_symbol_function_graph,
    load_symbol_module_graph,
)
from codeintel.storage.gateway import StorageGateway


def _component_counts(graph: nx.Graph | nx.DiGraph) -> tuple[int, int]:
    """
    Return weak component count and SCC/connected component count.

    Parameters
    ----------
    graph : nx.Graph | nx.DiGraph
        Graph to analyze; directed graphs compute weak and strongly connected components.

    Returns
    -------
    tuple[int, int]
        Weak component count and SCC/connected component count.
    """
    if isinstance(graph, nx.DiGraph):
        weak_count = nx.number_weakly_connected_components(graph)
        scc_count = sum(1 for _ in nx.strongly_connected_components(graph))
    else:
        weak_count = nx.number_connected_components(graph)
        scc_count = weak_count
    return weak_count, scc_count


def _avg_clustering(graph: nx.Graph | nx.DiGraph) -> float:
    undirected = graph.to_undirected()
    return nx.average_clustering(undirected) if undirected.number_of_nodes() > 0 else 0.0


def _diameter_and_spl(graph: nx.Graph | nx.DiGraph) -> tuple[float | None, float | None]:
    if graph.number_of_nodes() == 0:
        return None, None
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    if not components:
        return None, None
    largest = undirected.subgraph(max(components, key=len)).copy()
    try:
        diameter = float(approximation.diameter(largest))
    except NetworkXError:
        diameter = None
    try:
        avg_spl = float(nx.average_shortest_path_length(largest))
    except NetworkXError:
        avg_spl = None
    return diameter, avg_spl


def compute_graph_stats(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """
    Populate analytics.graph_stats for call/import graphs.

    Parameters
    ----------
    gateway :
        StorageGateway providing the DuckDB connection for source reads and destination writes.
    repo : str
        Repository identifier anchoring the metrics.
    commit : str
        Commit hash anchoring the metrics snapshot.
    """
    con = gateway.con
    ensure_schema(con, "analytics.graph_stats")
    graphs = {
        "call_graph": load_call_graph(gateway, repo, commit),
        "import_graph": load_import_graph(gateway, repo, commit),
        "symbol_module_graph": load_symbol_module_graph(gateway, repo, commit),
        "symbol_function_graph": load_symbol_function_graph(gateway, repo, commit),
    }
    config_bipartite = load_config_module_bipartite(gateway, repo, commit)
    if config_bipartite.number_of_nodes() > 0:
        keys = {n for n, d in config_bipartite.nodes(data=True) if d.get("bipartite") == 0}
        modules = set(config_bipartite) - keys
        graphs["config_key_projection"] = (
            nx.bipartite.weighted_projected_graph(config_bipartite, keys)
            if len(keys) > 1
            else nx.Graph()
        )
        graphs["config_module_projection"] = (
            nx.bipartite.weighted_projected_graph(config_bipartite, modules)
            if len(modules) > 1
            else nx.Graph()
        )
    now = datetime.now(UTC)
    rows: list[tuple[object, ...]] = []

    for name, graph in graphs.items():
        weak_count, scc_count = _component_counts(graph)
        diameter, avg_spl = _diameter_and_spl(graph)
        rows.append(
            (
                name,
                repo,
                commit,
                graph.number_of_nodes(),
                graph.number_of_edges(),
                weak_count,
                scc_count,
                _avg_clustering(graph),
                diameter,
                avg_spl,
                now,
            )
        )

    con.execute(
        "DELETE FROM analytics.graph_stats WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.graph_stats (
                graph_name, repo, commit, node_count, edge_count,
                weak_component_count, scc_count, avg_clustering,
                diameter_estimate, avg_shortest_path_estimate, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
