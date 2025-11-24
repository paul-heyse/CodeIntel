"""Config bipartite/projection graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import networkx as nx
from networkx.algorithms import community

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_config_module_bipartite
from codeintel.storage.gateway import StorageGateway

MAX_BETWEENNESS_NODES = 1000
MAX_COMMUNITY_NODES = 5000


@dataclass(frozen=True)
class ProjectionMetrics:
    """Centrality bundle for a projected bipartite graph."""

    deg: dict[tuple[str, str], int]
    wdeg: dict[tuple[str, str], float]
    bet: dict[tuple[str, str], float]
    clo: dict[tuple[str, str], float]
    comm_map: dict[tuple[str, str], int]


def _clear_config_tables(gateway: StorageGateway, repo: str, commit: str) -> None:
    con = gateway.con
    con.execute(
        "DELETE FROM analytics.config_graph_metrics_keys WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_graph_metrics_modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_projection_key_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.config_projection_module_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )


def _build_projection(graph: nx.Graph, nodes: set[tuple[str, str]]) -> nx.Graph:
    if not nodes:
        return nx.Graph()
    proj = nx.Graph()
    proj.add_nodes_from(nodes)
    if len(nodes) > 1:
        return nx.bipartite.weighted_projected_graph(graph, nodes)
    return proj


def _projection_metrics(proj: nx.Graph) -> ProjectionMetrics:
    communities = []
    if 0 < proj.number_of_nodes() <= MAX_COMMUNITY_NODES:
        communities = list(community.asyn_lpa_communities(proj, weight="weight"))
    comm_map = {node: idx for idx, comm in enumerate(communities) for node in comm}

    bet = {}
    if proj.number_of_nodes() > 0:
        k = min(proj.number_of_nodes(), MAX_BETWEENNESS_NODES)
        bet = nx.betweenness_centrality(
            proj, weight="weight", k=k if k < proj.number_of_nodes() else None
        )
    clo = nx.closeness_centrality(proj) if proj.number_of_nodes() > 0 else {}
    deg = {node: int(sum(1 for _ in proj.neighbors(node))) for node in proj.nodes}
    wdeg = {
        node: float(sum(data.get("weight", 1.0) for _, _, data in proj.edges(node, data=True)))
        for node in proj.nodes
    }
    return ProjectionMetrics(deg=deg, wdeg=wdeg, bet=bet, clo=clo, comm_map=comm_map)


def _projection_rows(
    *,
    proj: nx.Graph,
    repo: str,
    commit: str,
    now: datetime,
    metrics: ProjectionMetrics,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    node_rows = [
        (
            repo,
            commit,
            node[1],
            metrics.deg.get(node, 0),
            metrics.wdeg.get(node, 0.0),
            metrics.bet.get(node, 0.0),
            metrics.clo.get(node, 0.0),
            metrics.comm_map.get(node),
            now,
        )
        for node in proj.nodes
    ]
    edge_rows = [
        (
            repo,
            commit,
            src[1],
            dst[1],
            float(data.get("weight", 1.0)),
            now,
        )
        for src, dst, data in proj.edges(data=True)
    ]
    return cast("list[tuple[object, ...]]", node_rows), cast("list[tuple[object, ...]]", edge_rows)


def compute_config_graph_metrics(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Compute metrics for config keys/modules and their projections."""
    con = gateway.con
    ensure_schema(con, "analytics.config_graph_metrics_keys")
    ensure_schema(con, "analytics.config_graph_metrics_modules")
    ensure_schema(con, "analytics.config_projection_key_edges")
    ensure_schema(con, "analytics.config_projection_module_edges")

    graph = load_config_module_bipartite(gateway, repo, commit)
    if graph.number_of_nodes() == 0:
        _clear_config_tables(gateway, repo, commit)
        return
    now = datetime.now(UTC)

    keys = {node for node, data in graph.nodes(data=True) if data.get("bipartite") == 0}
    modules = set(graph) - keys
    if len(keys) == 0 or len(modules) == 0:
        _clear_config_tables(gateway, repo, commit)
        return

    key_rows: list[tuple[object, ...]] = []
    module_rows: list[tuple[object, ...]] = []
    key_edges: list[tuple[object, ...]] = []
    module_edges: list[tuple[object, ...]] = []

    if keys:
        key_proj = _build_projection(graph, keys)
        key_metrics = _projection_metrics(key_proj)
        key_rows, key_edges = _projection_rows(
            proj=key_proj,
            repo=repo,
            commit=commit,
            now=now,
            metrics=key_metrics,
        )

    if modules:
        module_proj = _build_projection(graph, modules)
        module_metrics = _projection_metrics(module_proj)
        module_rows, module_edges = _projection_rows(
            proj=module_proj,
            repo=repo,
            commit=commit,
            now=now,
            metrics=module_metrics,
        )

    _clear_config_tables(gateway, repo, commit)

    if key_rows:
        con.executemany(
            """
            INSERT INTO analytics.config_graph_metrics_keys (
                repo, commit, config_key, degree, weighted_degree,
                betweenness, closeness, community_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            key_rows,
        )
    if module_rows:
        con.executemany(
            """
            INSERT INTO analytics.config_graph_metrics_modules (
                repo, commit, module, degree, weighted_degree,
                betweenness, closeness, community_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            module_rows,
        )
    if key_edges:
        con.executemany(
            """
            INSERT INTO analytics.config_projection_key_edges (
                repo, commit, src_key, dst_key, weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            key_edges,
        )
    if module_edges:
        con.executemany(
            """
            INSERT INTO analytics.config_projection_module_edges (
                repo, commit, src_module, dst_module, weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            module_edges,
        )
