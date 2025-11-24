"""Symbol-coupling graph metrics for modules and functions."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import networkx as nx
from networkx.algorithms import structuralholes

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_symbol_function_graph, load_symbol_module_graph
from codeintel.storage.gateway import StorageGateway

MAX_BETWEENNESS_NODES = 1000
MAX_COMMUNITY_NODES = 5000


def _centrality_bundle(
    graph: nx.Graph, weight: str | None = None
) -> tuple[
    Mapping[Any, float],
    Mapping[Any, float],
    Mapping[Any, float],
    Mapping[Any, float],
    Mapping[Any, int | None],
    Mapping[Any, float | None],
    Mapping[Any, float | None],
    Mapping[Any, int],
    Mapping[Any, int],
    Mapping[Any, int],
]:
    node_count = graph.number_of_nodes()
    bet: Mapping[Any, float] = {}
    if node_count > 0:
        k = min(node_count, MAX_BETWEENNESS_NODES)
        bet = nx.betweenness_centrality(graph, k=k if k < node_count else None, weight=weight)
    clo = nx.closeness_centrality(graph) if graph.number_of_nodes() > 0 else {}
    har = nx.harmonic_centrality(graph) if graph.number_of_nodes() > 0 else {}
    eig = (
        nx.eigenvector_centrality(graph, max_iter=200, weight=weight)
        if graph.number_of_nodes() > 0
        else {}
    )
    k_core = nx.core_number(graph) if graph.number_of_nodes() > 0 else {}
    constraint_vals = (
        structuralholes.constraint(graph, weight=weight) if graph.number_of_nodes() > 0 else {}
    )
    eff_size = (
        structuralholes.effective_size(graph, weight=weight) if graph.number_of_nodes() > 0 else {}
    )
    community_id: dict[Any, int] = {}
    if node_count > 0 and node_count <= MAX_COMMUNITY_NODES:
        communities = list(nx.algorithms.community.asyn_lpa_communities(graph, weight=weight))
        community_id = {node: idx for idx, comm in enumerate(communities) for node in comm}
    components = list(nx.connected_components(graph))
    comp_id = {node: idx for idx, comp in enumerate(components) for node in comp}
    comp_size = {node: len(comp) for comp in components for node in comp}
    return bet, clo, eig, har, k_core, constraint_vals, eff_size, community_id, comp_id, comp_size


def compute_symbol_graph_metrics_modules(
    gateway: StorageGateway, *, repo: str, commit: str
) -> None:
    """Populate analytics.symbol_graph_metrics_modules from module symbol coupling."""
    con = gateway.con
    ensure_schema(con, "analytics.symbol_graph_metrics_modules")
    graph = load_symbol_module_graph(gateway, repo, commit)
    if graph.number_of_nodes() == 0:
        con.execute(
            "DELETE FROM analytics.symbol_graph_metrics_modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        return
    bet, clo, eig, har, k_core, constraint_vals, eff_size, community_id, comp_id, comp_size = (
        _centrality_bundle(graph, weight="weight")
    )
    now = datetime.now(UTC)
    rows = [
        (
            repo,
            commit,
            module,
            bet.get(module, 0.0),
            clo.get(module, 0.0),
            eig.get(module, 0.0),
            har.get(module, 0.0),
            k_core.get(module),
            constraint_vals.get(module, 0.0),
            eff_size.get(module, 0.0),
            community_id.get(module),
            comp_id.get(module),
            comp_size.get(module),
            now,
        )
        for module in graph.nodes
    ]
    con.execute(
        "DELETE FROM analytics.symbol_graph_metrics_modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.symbol_graph_metrics_modules (
                repo, commit, module,
                symbol_betweenness, symbol_closeness, symbol_eigenvector, symbol_harmonic,
                symbol_k_core, symbol_constraint, symbol_effective_size,
                symbol_community_id, symbol_component_id, symbol_component_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def compute_symbol_graph_metrics_functions(
    gateway: StorageGateway, *, repo: str, commit: str
) -> None:
    """Populate analytics.symbol_graph_metrics_functions from function symbol coupling."""
    con = gateway.con
    ensure_schema(con, "analytics.symbol_graph_metrics_functions")
    graph = load_symbol_function_graph(gateway, repo, commit)
    if graph.number_of_nodes() == 0:
        con.execute(
            "DELETE FROM analytics.symbol_graph_metrics_functions WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        return
    bet, clo, eig, har, k_core, constraint_vals, eff_size, community_id, comp_id, comp_size = (
        _centrality_bundle(graph, weight="weight")
    )
    now = datetime.now(UTC)
    rows = [
        (
            repo,
            commit,
            Decimal(goid),
            bet.get(goid, 0.0),
            clo.get(goid, 0.0),
            eig.get(goid, 0.0),
            har.get(goid, 0.0),
            k_core.get(goid),
            constraint_vals.get(goid, 0.0),
            eff_size.get(goid, 0.0),
            community_id.get(goid),
            comp_id.get(goid),
            comp_size.get(goid),
            now,
        )
        for goid in graph.nodes
    ]
    con.execute(
        "DELETE FROM analytics.symbol_graph_metrics_functions WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    if rows:
        con.executemany(
            """
            INSERT INTO analytics.symbol_graph_metrics_functions (
                repo, commit, function_goid_h128,
                symbol_betweenness, symbol_closeness, symbol_eigenvector, symbol_harmonic,
                symbol_k_core, symbol_constraint, symbol_effective_size,
                symbol_community_id, symbol_component_id, symbol_component_size, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
