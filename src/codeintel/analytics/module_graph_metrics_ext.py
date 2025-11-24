"""Extended module-level import graph metrics using NetworkX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import networkx as nx
from networkx.algorithms import structuralholes

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_import_graph
from codeintel.storage.gateway import StorageGateway

CENTRALITY_SAMPLE_LIMIT = 500
RICH_CLUB_PERCENTILE = 0.1


@dataclass(frozen=True)
class ModuleCentralities:
    """Centrality metrics computed on the import graph."""

    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    harmonic: dict[Any, float]
    eigen: dict[Any, float]
    k_core: dict[Any, int]
    constraint_vals: dict[Any, float]
    eff_size: dict[Any, float]


@dataclass(frozen=True)
class ModuleComponentData:
    """Component and community metadata for modules."""

    community_id: dict[Any, int]
    comp_id: dict[Any, int]
    comp_size: dict[Any, int]
    scc_id: dict[Any, int]
    scc_size: dict[Any, int]


def _centralities(graph: nx.DiGraph, undirected: nx.Graph) -> ModuleCentralities:
    betweenness = nx.betweenness_centrality(
        graph,
        weight="weight",
        k=min(CENTRALITY_SAMPLE_LIMIT, graph.number_of_nodes())
        if graph.number_of_nodes() > CENTRALITY_SAMPLE_LIMIT
        else None,
    )
    closeness = (
        cast("dict[Any, float]", nx.closeness_centrality(graph))
        if graph.number_of_nodes() > 0
        else {}
    )
    harmonic = (
        cast("dict[Any, float]", nx.harmonic_centrality(graph))
        if graph.number_of_nodes() > 0
        else {}
    )
    eigen = (
        nx.eigenvector_centrality(undirected, max_iter=200, weight="weight")
        if undirected.number_of_nodes() > 0
        else {}
    )
    k_core = nx.core_number(undirected) if undirected.number_of_nodes() > 0 else {}
    constraint_vals = (
        structuralholes.constraint(undirected, weight="weight")
        if undirected.number_of_nodes() > 0
        else {}
    )
    eff_size = (
        structuralholes.effective_size(undirected, weight="weight")
        if undirected.number_of_nodes() > 0
        else {}
    )
    return ModuleCentralities(
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigen=eigen,
        k_core=k_core,
        constraint_vals=constraint_vals,
        eff_size=eff_size,
    )


def _component_and_community(graph: nx.DiGraph, undirected: nx.Graph) -> ModuleComponentData:
    community_id: dict[Any, int] = {}
    if undirected.number_of_nodes() > 0:
        communities = list(
            nx.algorithms.community.asyn_lpa_communities(undirected, weight="weight")
        )
        for idx, community in enumerate(communities):
            for node in community:
                community_id[node] = idx

    weak_components = list(nx.weakly_connected_components(graph))
    comp_id = {node: idx for idx, comp in enumerate(weak_components) for node in comp}
    comp_size = {node: len(comp) for comp in weak_components for node in comp}
    sccs = list(nx.strongly_connected_components(graph))
    scc_id = {node: idx for idx, comp in enumerate(sccs) for node in comp}
    scc_size = {node: len(comp) for comp in sccs for node in comp}
    return ModuleComponentData(
        community_id=community_id,
        comp_id=comp_id,
        comp_size=comp_size,
        scc_id=scc_id,
        scc_size=scc_size,
    )


def compute_graph_metrics_modules_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Populate analytics.graph_metrics_modules_ext with richer import metrics."""
    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_modules_ext")
    graph = load_import_graph(gateway, repo, commit)
    simple_graph = graph.copy()
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()
    now = datetime.now(UTC)

    centralities = _centralities(simple_graph, undirected)
    components = _component_and_community(simple_graph, undirected)
    degree_view = cast("nx.classes.reportviews.DegreeView", simple_graph.degree)
    degree_map = {node: int(degree_view[node]) for node in simple_graph.nodes}
    degree_cutoff = 0
    if degree_map:
        sorted_nodes = sorted(degree_map.values(), reverse=True)
        idx = max(0, int(len(sorted_nodes) * RICH_CLUB_PERCENTILE) - 1)
        if sorted_nodes:
            degree_cutoff = sorted_nodes[idx] if idx < len(sorted_nodes) else sorted_nodes[-1]

    rows: list[tuple[object, ...]] = [
        (
            repo,
            commit,
            module,
            centralities.betweenness.get(module, 0.0),
            centralities.closeness.get(module, 0.0),
            centralities.eigen.get(module, 0.0),
            centralities.harmonic.get(module, 0.0),
            centralities.k_core.get(module),
            centralities.constraint_vals.get(module),
            centralities.eff_size.get(module),
            degree_map.get(module, 0) >= degree_cutoff if degree_cutoff > 0 else False,
            centralities.k_core.get(module),
            components.community_id.get(module),
            components.comp_id.get(module),
            components.comp_size.get(module),
            components.scc_id.get(module),
            components.scc_size.get(module),
            now,
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
