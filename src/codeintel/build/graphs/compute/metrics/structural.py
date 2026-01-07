"""Pure structural metric computation functions.

This module provides stateless functions for computing structural graph
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.graphs.compute.metrics.community import detect_communities_greedy
from codeintel.build.graphs.compute.metrics.paths import count_simple_paths
from codeintel.build.graphs.compute.metrics.types import (
    StructuralMetrics as StructuralSummary,
)
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    clustering_by_id,
    constraint_by_id,
    core_number_by_id,
    effective_size_by_id,
    ensure_store,
    triangles_by_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural graph features for a node.

    Attributes
    ----------
    clustering
        Local clustering coefficient.
    triangles
        Number of triangles the node participates in.
    core_number
        k-core number (maximum k for which node is in k-core).
    constraint
        Burt's constraint (structural holes).
    effective_size
        Effective size of ego network (structural holes).
    """

    clustering: float
    triangles: int
    core_number: int
    constraint: float
    effective_size: float


def compute_clustering_coefficient(
    graph: GraphInput,
    *,
    weight: str | None = None,
) -> dict[Any, float]:
    """Compute local clustering coefficient for all nodes.

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    weight
        Edge attribute to use as weight (None for unweighted).

    Returns
    -------
    dict[Any, float]
        Node to clustering coefficient mapping.
    """
    store = ensure_store(graph, weight=weight)
    if store.graph.num_nodes() == 0:
        return {}
    return clustering_by_id(store, weight=weight)


def compute_triangles(
    graph: GraphInput,
) -> dict[Any, int]:
    """Compute triangle count for all nodes.

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, int]
        Node to triangle count mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return triangles_by_id(store)


def compute_core_number(
    graph: GraphInput,
) -> dict[Any, int]:
    """Compute k-core number for all nodes.

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, int]
        Node to core number mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return core_number_by_id(store)


def compute_constraint(
    graph: GraphInput,
) -> dict[Any, float]:
    """Compute Burt's constraint for all nodes.

    Constraint measures how much a node's connections are to others who
    are themselves connected (indicating structural holes when low).

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, float]
        Node to constraint mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    try:
        constraint_result = constraint_by_id(store)
        return {
            node: 0.0 if math.isnan(val) else float(val) for node, val in constraint_result.items()
        }
    except (TypeError, ValueError):
        log.warning("Cannot compute constraint; returning zeros")
        return dict.fromkeys(store.node_ids(), 0.0)


def compute_effective_size(
    graph: GraphInput,
) -> dict[Any, float]:
    """Compute effective size of ego network for all nodes.

    Effective size measures the non-redundant portion of a node's
    neighborhood (related to structural holes).

    For directed graphs, computes on the undirected view.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, float]
        Node to effective size mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    try:
        effective_size_result = effective_size_by_id(store)
        return {node: float(val) for node, val in effective_size_result.items()}
    except (TypeError, ValueError):
        log.warning("Cannot compute effective size; returning zeros")
        return dict.fromkeys(store.node_ids(), 0.0)


def compute_all_structural(
    graph: GraphInput,
) -> dict[Any, StructuralMetrics]:
    """Compute all structural metrics for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, StructuralMetrics]
        Node to structural metrics mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}

    clustering = compute_clustering_coefficient(store)
    triangles = compute_triangles(store)
    core_number = compute_core_number(store)
    constraint = compute_constraint(store)
    effective_size = compute_effective_size(store)

    return {
        node: StructuralMetrics(
            clustering=clustering.get(node, 0.0),
            triangles=triangles.get(node, 0),
            core_number=core_number.get(node, 0),
            constraint=constraint.get(node, 0.0),
            effective_size=effective_size.get(node, 0.0),
        )
        for node in store.node_ids()
    }


def structural_metrics(
    graph: GraphInput,
    *,
    weight: str | None = "weight",
    community_limit: int | None = None,
) -> StructuralSummary:
    """Compute structural metrics for undirected graphs.

    Returns
    -------
    StructuralSummary
        Structural metric summary for the graph.
    """
    store = ensure_store(graph, weight=weight)
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return StructuralSummary(
            clustering={},
            triangles={},
            core_number={},
            constraint={},
            effective_size={},
            community_id={},
        )

    clustering = compute_clustering_coefficient(store, weight=weight)
    triangles = compute_triangles(store)
    core_number = compute_core_number(store)
    constraint_vals = compute_constraint(store)
    effective_size_vals = compute_effective_size(store)

    community_id_map: dict[Any, int] = {}
    if community_limit is None or node_count <= community_limit:
        community_id_map = detect_communities_greedy(store, weight=weight)

    return StructuralSummary(
        clustering=clustering,
        triangles=triangles,
        core_number=core_number,
        constraint=constraint_vals,
        effective_size=effective_size_vals,
        community_id=community_id_map,
    )


def bounded_simple_path_count(
    graph: GraphInput,
    sources: Iterable[Any],
    targets: Iterable[Any],
    *,
    max_paths: int,
    cutoff: int,
) -> int:
    """Count simple paths between sources and targets with hard limits.

    Returns
    -------
    int
        Count of simple paths, bounded by max_paths and cutoff.
    """
    return count_simple_paths(graph, sources, targets, max_paths=max_paths, cutoff=cutoff)


__all__ = [
    "StructuralMetrics",
    "bounded_simple_path_count",
    "compute_all_structural",
    "compute_clustering_coefficient",
    "compute_constraint",
    "compute_core_number",
    "compute_effective_size",
    "compute_triangles",
    "structural_metrics",
]
