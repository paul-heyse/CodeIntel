"""Shared data classes for graph metric results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NeighborStats:
    """Neighbor and edge count summaries for directed graphs."""

    in_neighbors: dict[Any, set[Any]]
    out_neighbors: dict[Any, set[Any]]
    in_counts: dict[Any, int]
    out_counts: dict[Any, int]


@dataclass(frozen=True)
class CentralityBundle:
    """Centrality metrics for a directed graph."""

    pagerank: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    harmonic: dict[Any, float]
    eigenvector: dict[Any, float]


@dataclass(frozen=True)
class ComponentBundle:
    """Component metadata for directed graphs."""

    component_id: dict[Any, int]
    component_size: dict[Any, int]
    scc_id: dict[Any, int]
    scc_size: dict[Any, int]
    in_cycle: dict[Any, bool]
    layer: dict[Any, int]


@dataclass(frozen=True)
class ProjectionMetrics:
    """Centrality bundle for projected bipartite graphs."""

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    clustering: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    community_id: dict[Any, int]


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural graph features for undirected graphs."""

    clustering: dict[Any, float]
    triangles: dict[Any, int]
    core_number: dict[Any, int]
    constraint: dict[Any, float]
    effective_size: dict[Any, float]
    community_id: dict[Any, int]


@dataclass(frozen=True)
class GlobalGraphStats:
    """Whole-graph summary statistics shared across analytics modules."""

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
    """Degree mappings for bipartite graphs."""

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    primary_degree_centrality: dict[Any, float]
    secondary_degree_centrality: dict[Any, float]


@dataclass(frozen=True)
class DominanceMetrics:
    """Dominator tree metrics for control-flow graphs."""

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
