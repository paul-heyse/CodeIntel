"""Pure coupling metric computation functions.

This module provides stateless functions for computing coupling metrics
and community detection without any database or file I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import networkx as nx

try:
    from networkx.algorithms.community import (
        label_propagation_communities,
        louvain_communities,
    )
except ImportError:
    label_propagation_communities = None
    louvain_communities = None


@dataclass(frozen=True)
class CouplingMetrics:
    """Coupling metrics for a node.

    Attributes
    ----------
    afferent
        Afferent coupling (incoming dependencies).
    efferent
        Efferent coupling (outgoing dependencies).
    instability
        Instability ratio (efferent / total).
    """

    afferent: int
    efferent: int
    instability: float


@dataclass(frozen=True)
class Community:
    """A detected community/cluster.

    Attributes
    ----------
    community_id
        Community identifier.
    nodes
        Nodes in the community.
    size
        Number of nodes.
    """

    community_id: int
    nodes: frozenset[Any]
    size: int


def compute_coupling(graph: nx.DiGraph) -> dict[Any, CouplingMetrics]:
    """Compute coupling metrics for all nodes.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    dict[Any, CouplingMetrics]
        Node to coupling metrics mapping.

    Examples
    --------
    >>> g = nx.DiGraph([(1, 2), (1, 3), (4, 1)])
    >>> metrics = compute_coupling(g)
    >>> metrics[1].efferent
    2
    >>> metrics[1].afferent
    1
    """
    result: dict[Any, CouplingMetrics] = {}
    for node in graph.nodes():
        # Cast needed due to imprecise NetworkX stubs
        afferent = cast("int", graph.in_degree(node))
        efferent = cast("int", graph.out_degree(node))
        total = afferent + efferent
        instability = efferent / total if total > 0 else 0.0
        result[node] = CouplingMetrics(
            afferent=afferent,
            efferent=efferent,
            instability=instability,
        )
    return result


def compute_abstractness(
    _node: object,
    abstract_count: int,
    total_count: int,
) -> float:
    """Compute abstractness for a module.

    Parameters
    ----------
    _node
        Module identifier (unused, kept for signature compatibility).
    abstract_count
        Number of abstract classes/interfaces.
    total_count
        Total number of classes.

    Returns
    -------
    float
        Abstractness ratio.
    """
    if total_count == 0:
        return 0.0
    return abstract_count / total_count


def compute_distance_from_main_sequence(
    coupling: CouplingMetrics,
    abstractness: float,
) -> float:
    """Compute distance from main sequence.

    The main sequence is A + I = 1, where A is abstractness and I is instability.
    Distance measures how far a module is from this ideal.

    Parameters
    ----------
    coupling
        Coupling metrics.
    abstractness
        Abstractness ratio.

    Returns
    -------
    float
        Distance from main sequence.
    """
    return abs(abstractness + coupling.instability - 1.0)


def detect_communities_louvain(graph: nx.Graph) -> list[Community]:
    """Detect communities using the Louvain algorithm.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[Community]
        Detected communities.
    """
    if graph.number_of_nodes() == 0:
        return []

    if louvain_communities is None:
        partitions = list(nx.connected_components(graph))
    else:
        partitions = louvain_communities(graph)

    return [
        Community(
            community_id=idx,
            nodes=frozenset(partition),
            size=len(partition),
        )
        for idx, partition in enumerate(partitions)
    ]


def detect_communities_label_propagation(graph: nx.Graph) -> list[Community]:
    """Detect communities using label propagation.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[Community]
        Detected communities.
    """
    if graph.number_of_nodes() == 0:
        return []

    if label_propagation_communities is None:
        partitions = list(nx.connected_components(graph))
    else:
        partitions = list(label_propagation_communities(graph))
    return [
        Community(
            community_id=idx,
            nodes=frozenset(partition),
            size=len(partition),
        )
        for idx, partition in enumerate(partitions)
    ]


def compute_modularity(
    graph: nx.Graph,
    communities: Sequence[Community],
) -> float:
    """Compute modularity score for a community partition.

    Parameters
    ----------
    graph
        Undirected graph.
    communities
        Community partition.

    Returns
    -------
    float
        Modularity score (-0.5 to 1.0).
    """
    if graph.number_of_nodes() == 0 or not communities:
        return 0.0

    partition = [set(c.nodes) for c in communities]
    return nx.community.modularity(graph, partition)


def compute_clustering_coefficient(graph: nx.Graph) -> dict[Any, float]:
    """Compute local clustering coefficient for all nodes.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    dict[Any, float]
        Node to clustering coefficient mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    # nx.clustering(graph) returns dict[node, float] when called with a graph
    # The stubs are imprecise, showing it could return float for single-node calls
    clustering_result = nx.clustering(graph)
    return cast("dict[Any, float]", clustering_result)


def compute_average_clustering(graph: nx.Graph) -> float:
    """Compute average clustering coefficient.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    float
        Average clustering coefficient.
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    return nx.average_clustering(graph)


def find_hub_nodes(
    graph: nx.Graph | nx.DiGraph,
    threshold_ratio: float = 0.1,
    min_degree: int = 5,
) -> list[Any]:
    """Find hub nodes with high connectivity.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    threshold_ratio
        Minimum ratio of max degree to be a hub.
    min_degree
        Minimum degree to be considered.

    Returns
    -------
    list[Any]
        Hub nodes.
    """
    if graph.number_of_nodes() == 0:
        return []

    # Build degree mapping from the graph
    # Use len(neighbors) to avoid NetworkX stub issues with graph.degree()
    degrees: dict[Any, int] = {}
    for node in graph.nodes():
        degrees[node] = len(list(graph.neighbors(node)))
    if not degrees:
        return []

    max_degree = max(degrees.values())
    threshold = max(min_degree, int(float(max_degree) * threshold_ratio))

    return [node for node, degree in degrees.items() if degree >= threshold]


def find_boundary_nodes(
    graph: nx.Graph,
    communities: Sequence[Community],
) -> list[Any]:
    """Find nodes at community boundaries.

    Parameters
    ----------
    graph
        Undirected graph.
    communities
        Community partition.

    Returns
    -------
    list[Any]
        Boundary nodes with neighbors in multiple communities.
    """
    if not communities:
        return []

    # Build node to community mapping
    node_community: dict[Any, int] = {}
    for comm in communities:
        for node in comm.nodes:
            node_community[node] = comm.community_id

    boundary: list[Any] = []
    for node in graph.nodes():
        neighbor_communities = {
            node_community.get(neighbor)
            for neighbor in graph.neighbors(node)
            if neighbor in node_community
        }
        if len(neighbor_communities) > 1:
            boundary.append(node)

    return boundary


def coupling_to_rows(
    metrics: Mapping[str, CouplingMetrics],
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    """Convert coupling metrics to row dictionaries.

    Parameters
    ----------
    metrics
        Module to coupling metrics mapping.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    list[dict[str, object]]
        Row dictionaries for persistence.
    """
    return [
        {
            "module": module,
            "repo": repo,
            "commit": commit,
            "afferent_coupling": m.afferent,
            "efferent_coupling": m.efferent,
            "instability": m.instability,
        }
        for module, m in metrics.items()
    ]


__all__ = [
    "Community",
    "CouplingMetrics",
    "compute_abstractness",
    "compute_average_clustering",
    "compute_clustering_coefficient",
    "compute_coupling",
    "compute_distance_from_main_sequence",
    "compute_modularity",
    "coupling_to_rows",
    "detect_communities_label_propagation",
    "detect_communities_louvain",
    "find_boundary_nodes",
    "find_hub_nodes",
]
