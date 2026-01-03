"""Pure structural metric computation functions.

This module provides stateless functions for computing structural graph
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
from networkx.exception import NetworkXError

from codeintel.build.graphs.compute.metrics.community import detect_communities_greedy
from codeintel.build.graphs.compute.metrics.paths import count_simple_paths
from codeintel.build.graphs.compute.metrics.types import (
    StructuralMetrics as StructuralSummary,
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
    graph: nx.Graph | nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
    clustering_result = cast("dict[Any, float]", nx.clustering(work_graph, weight=weight))
    return {node: float(val) for node, val in clustering_result.items()}


def compute_triangles(
    graph: nx.Graph | nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
    triangles_result = cast("dict[Any, int]", nx.triangles(work_graph))
    return {node: int(val) for node, val in triangles_result.items()}


def compute_core_number(
    graph: nx.Graph | nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
    core_result: dict[Any, int] = nx.core_number(work_graph)
    return {node: int(val) for node, val in core_result.items()}


def compute_constraint(
    graph: nx.Graph | nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    try:
        constraint_result: dict[Any, float] = nx.constraint(work_graph)

        return {
            node: 0.0 if math.isnan(val) else float(val) for node, val in constraint_result.items()
        }
    except NetworkXError:
        log.warning("Cannot compute constraint; returning zeros")
        return dict.fromkeys(graph.nodes(), 0.0)


def compute_effective_size(
    graph: nx.Graph | nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    try:
        effective_size_result: dict[Any, float] = nx.effective_size(work_graph)
        return {node: float(val) for node, val in effective_size_result.items()}
    except NetworkXError:
        log.warning("Cannot compute effective size; returning zeros")
        return dict.fromkeys(graph.nodes(), 0.0)


def compute_all_structural(
    graph: nx.Graph | nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    clustering = compute_clustering_coefficient(graph)
    triangles = compute_triangles(graph)
    core_number = compute_core_number(graph)
    constraint = compute_constraint(graph)
    effective_size = compute_effective_size(graph)

    return {
        node: StructuralMetrics(
            clustering=clustering.get(node, 0.0),
            triangles=triangles.get(node, 0),
            core_number=core_number.get(node, 0),
            constraint=constraint.get(node, 0.0),
            effective_size=effective_size.get(node, 0.0),
        )
        for node in graph.nodes()
    }


def structural_metrics(
    graph: nx.Graph,
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
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return StructuralSummary(
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
        community_id_map = detect_communities_greedy(graph, weight=weight)

    return StructuralSummary(
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
