"""Extended NetworkX-derived metrics for the call graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import networkx as nx

from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.nx_views import load_call_graph
from codeintel.storage.gateway import StorageGateway

CENTRALITY_SAMPLE_LIMIT = 500
EIGEN_MAX_ITER = 200


@dataclass(frozen=True)
class FunctionCentralities:
    """Centrality metrics computed on the call graph."""

    betweenness: Mapping[Any, float]
    closeness: Mapping[Any, float]
    harmonic: Mapping[Any, float]
    eigen: Mapping[Any, float]
    core_number: Mapping[Any, int]
    clustering: Mapping[Any, float]
    triangles: Mapping[Any, int]


@dataclass(frozen=True)
class ComponentData:
    """Connected component and articulation metadata for the call graph."""

    comp_id: dict[Any, int]
    comp_size: dict[Any, int]
    scc_id: dict[Any, int]
    scc_size: dict[Any, int]
    articulations: set[Any]
    bridge_incident: dict[Any, int]


def _to_decimal(value: int) -> Decimal:
    """
    Coerce integer GOIDs into DECIMAL for DuckDB writes.

    Parameters
    ----------
    value : int
        Integer GOID to convert.

    Returns
    -------
    Decimal
        Decimal representation for DuckDB storage.
    """
    return Decimal(value)


def _centralities(graph: nx.DiGraph, undirected: nx.Graph) -> FunctionCentralities:
    betweenness: Mapping[Any, float] = nx.betweenness_centrality(
        graph,
        k=min(CENTRALITY_SAMPLE_LIMIT, graph.number_of_nodes())
        if graph.number_of_nodes() > CENTRALITY_SAMPLE_LIMIT
        else None,
    )
    if graph.number_of_nodes() > 0:
        closeness: Mapping[Any, float] = nx.closeness_centrality(graph)
        harmonic: Mapping[Any, float] = nx.harmonic_centrality(graph)
    else:
        closeness = cast("Mapping[Any, float]", {})
        harmonic = cast("Mapping[Any, float]", {})

    if undirected.number_of_nodes() > 0:
        eigen: Mapping[Any, float] = nx.eigenvector_centrality(undirected, max_iter=EIGEN_MAX_ITER)
        core_number: Mapping[Any, int] = nx.core_number(undirected)
        clustering_val = nx.clustering(undirected)
        clustering: Mapping[Any, float] = cast(
            "Mapping[Any, float]", clustering_val if isinstance(clustering_val, dict) else {}
        )
        triangles_val = nx.triangles(undirected)
        triangles: Mapping[Any, int] = cast(
            "Mapping[Any, int]", triangles_val if isinstance(triangles_val, dict) else {}
        )
    else:
        eigen = cast("Mapping[Any, float]", {})
        core_number = cast("Mapping[Any, int]", {})
        clustering = cast("Mapping[Any, float]", {})
        triangles = cast("Mapping[Any, int]", {})

    return FunctionCentralities(
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigen=eigen,
        core_number=core_number,
        clustering=clustering,
        triangles=triangles,
    )


def _component_data(graph: nx.DiGraph, undirected: nx.Graph) -> ComponentData:
    weak_components = list(nx.weakly_connected_components(graph))
    comp_id: dict[Any, int] = {
        node: idx for idx, component in enumerate(weak_components) for node in component
    }
    comp_size: dict[Any, int] = {
        node: len(component) for component in weak_components for node in component
    }
    sccs = list(nx.strongly_connected_components(graph))
    scc_id: dict[Any, int] = {node: idx for idx, comp in enumerate(sccs) for node in comp}
    scc_size: dict[Any, int] = {node: len(comp) for idx, comp in enumerate(sccs) for node in comp}
    articulations = set(nx.articulation_points(undirected))
    bridge_incident: dict[Any, int] = dict.fromkeys(undirected.nodes, 0)
    for left, right in nx.bridges(undirected):
        bridge_incident[left] += 1
        bridge_incident[right] += 1
    return ComponentData(
        comp_id=comp_id,
        comp_size=comp_size,
        scc_id=scc_id,
        scc_size=scc_size,
        articulations=articulations,
        bridge_incident=bridge_incident,
    )


def compute_graph_metrics_functions_ext(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
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
    """
    con = gateway.con
    ensure_schema(con, "analytics.graph_metrics_functions_ext")
    graph = load_call_graph(gateway, repo, commit)
    simple_graph = graph.copy()
    simple_graph.remove_edges_from(nx.selfloop_edges(simple_graph))
    undirected = simple_graph.to_undirected()
    now = datetime.now(UTC)

    centralities = _centralities(simple_graph, undirected)
    components = _component_data(simple_graph, undirected)
    community_id: dict[Any, int] = {}
    if undirected.number_of_nodes() > 0:
        communities = list(
            nx.algorithms.community.asyn_lpa_communities(undirected, weight="weight")
        )
        for idx, community in enumerate(communities):
            for member in community:
                community_id[member] = idx

    rows: list[tuple[object, ...]] = []
    for node in simple_graph.nodes:
        goid = _to_decimal(node)
        ancestor_count = len(nx.ancestors(graph, node)) if graph.number_of_nodes() else 0
        descendant_count = len(nx.descendants(graph, node)) if graph.number_of_nodes() else 0
        rows.append(
            (
                repo,
                commit,
                goid,
                centralities.betweenness.get(node, 0.0),
                centralities.closeness.get(node, 0.0),
                centralities.eigen.get(node, 0.0),
                centralities.harmonic.get(node, 0.0),
                centralities.core_number.get(node),
                centralities.clustering.get(node, 0.0),
                int(centralities.triangles.get(node, 0)),
                node in components.articulations,
                None,  # placeholder for articulation impact
                components.bridge_incident.get(node, 0) > 0,
                components.comp_id.get(node),
                components.comp_size.get(node),
                components.scc_id.get(node),
                components.scc_size.get(node),
                ancestor_count,
                descendant_count,
                community_id.get(node),
                now,
            )
        )

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
