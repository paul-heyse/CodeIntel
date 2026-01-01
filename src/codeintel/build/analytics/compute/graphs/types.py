"""Data classes for graph metric results.

This module contains immutable data classes that represent computed
graph metrics. These types are used across the graph_primitives package
and by analytics consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NeighborStats:
    """Neighbor and edge count summaries for directed graphs.

    Attributes
    ----------
    in_neighbors
        Mapping from node to set of incoming neighbors.
    out_neighbors
        Mapping from node to set of outgoing neighbors.
    in_counts
        Mapping from node to weighted in-degree.
    out_counts
        Mapping from node to weighted out-degree.
    """

    in_neighbors: dict[Any, set[Any]]
    out_neighbors: dict[Any, set[Any]]
    in_counts: dict[Any, int]
    out_counts: dict[Any, int]


@dataclass(frozen=True)
class CentralityBundle:
    """Centrality metrics for a directed graph.

    Attributes
    ----------
    pagerank
        PageRank scores.
    betweenness
        Betweenness centrality scores.
    closeness
        Closeness centrality scores.
    harmonic
        Harmonic centrality scores.
    eigenvector
        Eigenvector centrality scores (may be empty if not computed).
    """

    pagerank: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    harmonic: dict[Any, float]
    eigenvector: dict[Any, float]


@dataclass(frozen=True)
class ComponentBundle:
    """Component metadata for directed graphs.

    Attributes
    ----------
    component_id
        Mapping from node to weak component ID.
    component_size
        Mapping from node to weak component size.
    scc_id
        Mapping from node to strongly connected component ID.
    scc_size
        Mapping from node to SCC size.
    in_cycle
        Mapping from node to whether it's in a cycle.
    layer
        Mapping from node to condensation layer.
    """

    component_id: dict[Any, int]
    component_size: dict[Any, int]
    scc_id: dict[Any, int]
    scc_size: dict[Any, int]
    in_cycle: dict[Any, bool]
    layer: dict[Any, int]


@dataclass(frozen=True)
class ProjectionMetrics:
    """Centrality bundle for projected bipartite graphs.

    Attributes
    ----------
    degree
        Unweighted degree per node.
    weighted_degree
        Weighted degree per node.
    clustering
        Clustering coefficient per node.
    betweenness
        Betweenness centrality per node.
    closeness
        Closeness centrality per node.
    community_id
        Community assignment per node.
    """

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    clustering: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    community_id: dict[Any, int]


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural graph features for undirected graphs.

    Attributes
    ----------
    clustering
        Clustering coefficient per node.
    triangles
        Triangle count per node.
    core_number
        K-core number per node.
    constraint
        Burt's constraint per node.
    effective_size
        Effective size per node.
    community_id
        Community assignment per node.
    """

    clustering: dict[Any, float]
    triangles: dict[Any, int]
    core_number: dict[Any, int]
    constraint: dict[Any, float]
    effective_size: dict[Any, float]
    community_id: dict[Any, int]


@dataclass(frozen=True)
class GlobalGraphStats:
    """Whole-graph summary statistics shared across analytics modules.

    Attributes
    ----------
    node_count
        Number of nodes in the graph.
    edge_count
        Number of edges in the graph.
    weak_component_count
        Number of weakly connected components.
    scc_count
        Number of strongly connected components.
    component_layers
        Number of layers in the condensation DAG.
    avg_clustering
        Average clustering coefficient.
    diameter_estimate
        Estimated graph diameter.
    avg_shortest_path_estimate
        Estimated average shortest path length.
    """

    node_count: int
    edge_count: int
    weak_component_count: int
    scc_count: int
    component_layers: int | None
    avg_clustering: float
    diameter_estimate: float | None
    avg_shortest_path_estimate: float | None


@dataclass(frozen=True)
class BipartiteDegrees:
    """Degree mappings for bipartite graphs.

    Attributes
    ----------
    degree
        Unweighted degree per node.
    weighted_degree
        Weighted degree per node.
    primary_degree_centrality
        Degree centrality for primary partition.
    secondary_degree_centrality
        Degree centrality for secondary partition.
    """

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    primary_degree_centrality: dict[Any, float]
    secondary_degree_centrality: dict[Any, float]


@dataclass(frozen=True)
class DominanceMetrics:
    """Dominator tree metrics for control-flow graphs.

    Attributes
    ----------
    depth
        Dominator tree depth per node.
    frontier_sizes
        Dominance frontier size per node.
    tree_height
        Height of the dominator tree.
    """

    depth: dict[Any, int]
    frontier_sizes: dict[Any, int]
    tree_height: int | None


__all__ = [
    "BipartiteDegrees",
    "CentralityBundle",
    "ComponentBundle",
    "DominanceMetrics",
    "GlobalGraphStats",
    "NeighborStats",
    "ProjectionMetrics",
    "StructuralMetrics",
]
