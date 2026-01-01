"""Pure computation for bipartite graph metrics.

This module provides functions to compute metrics for bipartite graphs,
including degree centrality and weighted projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
from networkx.algorithms import bipartite as nx_bipartite
from networkx.exception import NetworkXAlgorithmError


@dataclass(frozen=True)
class BipartiteDegreeMetrics:
    """Degree metrics for bipartite graph partitions.

    Attributes
    ----------
    degree
        Mapping of node to unweighted degree.
    weighted_degree
        Mapping of node to weighted degree.
    primary_degree_centrality
        Degree centrality for primary partition nodes.
    secondary_degree_centrality
        Degree centrality for secondary partition nodes.
    """

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    primary_degree_centrality: dict[Any, float]
    secondary_degree_centrality: dict[Any, float]


def compute_bipartite_degrees(
    graph: nx.Graph,
    primary: set[Any],
    secondary: set[Any],
    *,
    weight: str | None = "weight",
) -> BipartiteDegreeMetrics:
    """Compute degree metrics for bipartite graph partitions.

    Parameters
    ----------
    graph
        Bipartite graph to analyze.
    primary
        Set of primary partition nodes.
    secondary
        Set of secondary partition nodes.
    weight
        Edge attribute storing weight; defaults to "weight".

    Returns
    -------
    BipartiteDegreeMetrics
        Degree metrics for both partitions.

    Examples
    --------
    >>> g = nx.Graph()
    >>> g.add_edges_from([(1, "a"), (1, "b"), (2, "b")])
    >>> result = compute_bipartite_degrees(g, {1, 2}, {"a", "b"})
    >>> result.degree[1]
    2
    """
    degree: dict[Any, int] = {}
    weighted_degree: dict[Any, float] = {}

    unweighted_view = nx.degree(graph, weight=None)
    weighted_view = nx.degree(graph, weight=weight)

    for node, deg in unweighted_view:
        degree[node] = int(deg)
    for node, deg in weighted_view:
        weighted_degree[node] = float(deg)

    if not primary or not secondary:
        return BipartiteDegreeMetrics(
            degree=degree,
            weighted_degree=weighted_degree,
            primary_degree_centrality={},
            secondary_degree_centrality={},
        )

    primary_dc = nx_bipartite.degree_centrality(graph, secondary)
    secondary_dc = nx_bipartite.degree_centrality(graph, primary)

    return BipartiteDegreeMetrics(
        degree=degree,
        weighted_degree=weighted_degree,
        primary_degree_centrality={node: float(val) for node, val in primary_dc.items()},
        secondary_degree_centrality={node: float(val) for node, val in secondary_dc.items()},
    )


def compute_weighted_projection(
    bipartite_graph: nx.Graph,
    nodes: set[Any],
) -> nx.Graph | None:
    """Build a weighted projection graph from a bipartite partition.

    Parameters
    ----------
    bipartite_graph
        Bipartite graph to project.
    nodes
        Set of nodes in the partition to project onto.

    Returns
    -------
    nx.Graph | None
        Projected graph, or None if projection cannot be computed.

    Notes
    -----
    The projection fails and returns None if:
    - The nodes set is empty
    - The nodes are not a subset of the graph's nodes
    - The nodes set is equal to or larger than the entire graph

    Examples
    --------
    >>> g = nx.Graph()
    >>> g.add_edges_from([(1, "a"), (1, "b"), (2, "b")])
    >>> proj = compute_weighted_projection(g, {1, 2})
    >>> proj is not None
    True
    >>> proj.number_of_nodes()
    2
    """
    graph_nodes = bipartite_graph.number_of_nodes()
    if not nodes:
        return None
    graph_node_set = set(bipartite_graph)
    if not nodes.issubset(graph_node_set):
        return None
    if len(nodes) >= graph_nodes:
        return None
    try:
        return nx_bipartite.weighted_projected_graph(bipartite_graph, nodes)
    except NetworkXAlgorithmError:
        return None


__all__ = [
    "BipartiteDegreeMetrics",
    "compute_bipartite_degrees",
    "compute_weighted_projection",
]
