"""Pure community detection computation functions.

This module provides stateless functions for detecting communities
in graphs without any database or file I/O.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from networkx.algorithms import community as nx_community

log = logging.getLogger(__name__)


def detect_communities_greedy(
    graph: nx.Graph | nx.DiGraph,
    *,
    weight: str | None = None,
    resolution: float = 1.0,
) -> dict[Any, int]:
    """Detect communities using greedy modularity optimization.

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    weight
        Edge weight attribute (None for unweighted).
    resolution
        Resolution parameter for modularity.

    Returns
    -------
    dict[Any, int]
        Node to community ID mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    try:
        communities = nx_community.greedy_modularity_communities(
            work_graph,
            weight=weight,
            resolution=resolution,
        )
    except nx.NetworkXError as exc:
        log.warning("Community detection failed: %s", exc)
        # Fall back to each node in its own community
        return {node: idx for idx, node in enumerate(graph.nodes())}

    result: dict[Any, int] = {}
    for community_id, comm in enumerate(communities):
        for node in comm:
            result[node] = community_id
    return result


def detect_communities_louvain(
    graph: nx.Graph | nx.DiGraph,
    *,
    weight: str | None = None,
    resolution: float = 1.0,
    seed: int | None = None,
) -> dict[Any, int]:
    """Detect communities using Louvain algorithm.

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    weight
        Edge weight attribute (None for unweighted).
    resolution
        Resolution parameter for modularity.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict[Any, int]
        Node to community ID mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    try:
        communities = nx_community.louvain_communities(
            work_graph,
            weight=weight,
            resolution=resolution,
            seed=seed,
        )
    except nx.NetworkXError as exc:
        log.warning("Louvain community detection failed: %s", exc)
        return {node: idx for idx, node in enumerate(graph.nodes())}

    result: dict[Any, int] = {}
    for community_id, comm in enumerate(communities):
        for node in comm:
            result[node] = community_id
    return result


def detect_communities_label_propagation(
    graph: nx.Graph | nx.DiGraph,
) -> dict[Any, int]:
    """Detect communities using label propagation.

    This is faster than modularity-based methods but may be less stable.

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, int]
        Node to community ID mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    try:
        communities = nx_community.label_propagation_communities(work_graph)
    except nx.NetworkXError as exc:
        log.warning("Label propagation failed: %s", exc)
        return {node: idx for idx, node in enumerate(graph.nodes())}

    result: dict[Any, int] = {}
    for community_id, comm in enumerate(communities):
        for node in comm:
            result[node] = community_id
    return result


def compute_modularity(
    graph: nx.Graph | nx.DiGraph,
    communities: dict[Any, int],
    *,
    weight: str | None = None,
    resolution: float = 1.0,
) -> float:
    """Compute modularity of a community partition.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    communities
        Node to community ID mapping.
    weight
        Edge weight attribute (None for unweighted).
    resolution
        Resolution parameter.

    Returns
    -------
    float
        Modularity score.
    """
    if graph.number_of_nodes() == 0:
        return 0.0

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    # Group nodes by community
    community_sets: dict[int, set[Any]] = {}
    for node, comm_id in communities.items():
        if comm_id not in community_sets:
            community_sets[comm_id] = set()
        community_sets[comm_id].add(node)

    partition = list(community_sets.values())

    try:
        return float(
            nx_community.modularity(
                work_graph,
                partition,
                weight=weight,
                resolution=resolution,
            )
        )
    except (nx.NetworkXError, ZeroDivisionError):
        return 0.0


__all__ = [
    "compute_modularity",
    "detect_communities_greedy",
    "detect_communities_label_propagation",
    "detect_communities_louvain",
]
