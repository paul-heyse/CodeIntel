"""Structural graph metrics computation.

This module provides functions for computing structural metrics like
clustering, triangles, k-core, and structural holes on undirected graphs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx

from codeintel.analytics.compute.graphs.projections import community_ids
from codeintel.analytics.compute.graphs.types import StructuralMetrics
from codeintel.graphs.compute.metrics.paths import count_simple_paths
from codeintel.graphs.compute.metrics.structural import (
    compute_clustering_coefficient,
    compute_constraint,
    compute_core_number,
    compute_effective_size,
    compute_triangles,
)


def structural_metrics(
    graph: nx.Graph,
    *,
    weight: str | None = "weight",
    community_limit: int | None = None,
) -> StructuralMetrics:
    """Compute structural metrics for undirected graphs.

    Parameters
    ----------
    graph
        Undirected graph to evaluate.
    weight
        Edge attribute storing weight. Defaults to "weight".
    community_limit
        Optional cap on node count beyond which community detection is skipped.

    Returns
    -------
    StructuralMetrics
        Structural hole and core metrics.
    """
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return StructuralMetrics(
            clustering={},
            triangles={},
            core_number={},
            constraint={},
            effective_size={},
            community_id={},
        )

    clustering = compute_clustering_coefficient(graph)
    triangles = compute_triangles(graph)
    core_number = compute_core_number(graph)
    constraint_vals = compute_constraint(graph)
    effective_size_vals = compute_effective_size(graph)

    community_id_map: dict[Any, int] = {}
    if community_limit is None or node_count <= community_limit:
        community_id_map = community_ids(graph, weight=weight)

    return StructuralMetrics(
        clustering=clustering,
        triangles=triangles,
        core_number=core_number,
        constraint=constraint_vals,
        effective_size=effective_size_vals,
        community_id=community_id_map,
    )


def bounded_simple_path_count(
    graph: nx.DiGraph,
    sources: Iterable[Any],
    targets: Iterable[Any],
    *,
    max_paths: int,
    cutoff: int,
) -> int:
    """Count simple paths between sources and targets with hard limits.

    Delegates to graphs.compute.metrics.paths.count_simple_paths.

    Parameters
    ----------
    graph
        Directed graph to analyze.
    sources
        Iterable of source nodes.
    targets
        Iterable of target nodes.
    max_paths
        Maximum number of paths to count before stopping.
    cutoff
        Maximum path length to consider.

    Returns
    -------
    int
        Number of simple paths discovered up to the configured limit.
    """
    return count_simple_paths(graph, sources, targets, max_paths=max_paths, cutoff=cutoff)


__all__ = [
    "bounded_simple_path_count",
    "structural_metrics",
]
