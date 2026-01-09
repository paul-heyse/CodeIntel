"""Pure computation for graph statistics.

This module provides functions to compute summary statistics
for rustworkx graph stores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import rustworkx as rx

from codeintel.build.graphs.rx.algos import (
    GraphInput,
    connected_components_by_id,
    degree_by_id,
    ensure_store,
    graph_distance_matrix,
    graph_unweighted_average_shortest_path_length,
    in_degree_by_id,
    is_directed_acyclic,
    out_degree_by_id,
    strongly_connected_components_by_id,
    to_undirected_store,
    topological_generations_by_id,
    weakly_connected_components_by_id,
)
from codeintel.build.graphs.rx.condensation import condensation_store
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.store import RxGraphStore


@dataclass(frozen=True)
class GraphStatistics:
    """Summary statistics for a graph.

    Attributes
    ----------
    node_count
        Number of nodes in the graph.
    edge_count
        Number of edges in the graph.
    density
        Graph density (edges / possible edges).
    avg_in_degree
        Average in-degree.
    avg_out_degree
        Average out-degree.
    strongly_connected_components
        Number of strongly connected components.
    weakly_connected_components
        Number of weakly connected components.
    is_dag
        Whether the graph is a DAG.
    """

    node_count: int
    edge_count: int
    density: float
    avg_in_degree: float
    avg_out_degree: float
    strongly_connected_components: int
    weakly_connected_components: int
    is_dag: bool


def _undirected_graph(store: RxGraphStore) -> rx.PyGraph:
    if store.is_directed:
        message = "Expected an undirected graph store"
        raise ValueError(message)
    return cast("rx.PyGraph", store.graph)


def get_in_degrees(graph: GraphInput) -> list[tuple[Any, int]]:
    """Extract in-degree tuples from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[tuple[Any, int]]
        List of (node, in_degree) tuples for all nodes in the graph.

    Examples
    --------
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(1, 3, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> get_in_degrees(g)
    [(1, 0), (2, 1), (3, 2)]
    """
    return list(in_degree_by_id(graph).items())


def get_out_degrees(graph: GraphInput) -> list[tuple[Any, int]]:
    """Extract out-degree tuples from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[tuple[Any, int]]
        List of (node, out_degree) tuples for all nodes in the graph.

    Examples
    --------
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(1, 3, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> get_out_degrees(g)
    [(1, 2), (2, 1), (3, 0)]
    """
    return list(out_degree_by_id(graph).items())


def get_degrees(graph: GraphInput) -> list[tuple[Any, int]]:
    """Extract degree tuples from an undirected graph.

    Parameters
    ----------
    graph
        An undirected graph to analyze.

    Returns
    -------
    list[tuple[Any, int]]
        List of (node, degree) tuples for all nodes in the graph.

    Examples
    --------
    >>> g = RxGraphStore.undirected()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(1, 3, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> get_degrees(g)
    [(1, 2), (2, 2), (3, 2)]
    """
    return list(degree_by_id(graph).items())


def get_in_degree_values(graph: GraphInput) -> list[int]:
    """Extract just the in-degree values from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[int]
        List of in-degree values for all nodes (in node iteration order).
    """
    return [degree for _, degree in get_in_degrees(graph)]


def get_out_degree_values(graph: GraphInput) -> list[int]:
    """Extract just the out-degree values from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[int]
        List of out-degree values for all nodes (in node iteration order).
    """
    return [degree for _, degree in get_out_degrees(graph)]


def get_degree_values(graph: GraphInput) -> list[int]:
    """Extract just the degree values from an undirected graph.

    Parameters
    ----------
    graph
        An undirected graph to analyze.

    Returns
    -------
    list[int]
        List of degree values for all nodes (in node iteration order).
    """
    return [degree for _, degree in get_degrees(graph)]


def _component_sort_key(component: set[Any]) -> tuple[int, tuple[str, str]]:
    if not component:
        return (0, ("", ""))
    smallest = min(component, key=stable_key)
    return (len(component), stable_key(smallest))


def _largest_component(store: RxGraphStore) -> set[Any] | None:
    components = connected_components_by_id(store)
    if not components:
        return None
    return max(components, key=_component_sort_key)


def compute_diameter_estimate(graph: GraphInput) -> float | None:
    """Compute approximate diameter of the largest connected component.

    Parameters
    ----------
    graph
        Graph to analyze (directed or undirected).

    Returns
    -------
    float | None
        Approximate diameter of largest component, or None if empty.

    Examples
    --------
    >>> g = RxGraphStore.undirected()
    >>> g.add_weighted_edge(0, 1, weight=1.0)
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> g.add_weighted_edge(3, 4, weight=1.0)
    >>> compute_diameter_estimate(g)
    4.0
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return None
    work_store = to_undirected_store(store)
    largest = _largest_component(work_store)
    if largest is None:
        return None
    if len(largest) <= 1:
        return 0.0
    undirected_graph = _undirected_graph(work_store)
    ordered_indices = [
        work_store.id_to_index[node_id]
        for node_id in largest
        if node_id in work_store.id_to_index
    ]
    ordered_indices.sort(key=lambda idx: stable_key(work_store.index_to_id[idx]))
    subgraph, _node_map = undirected_graph.subgraph_with_nodemap(
        ordered_indices,
        preserve_attrs=True,
    )
    distances = graph_distance_matrix(subgraph)
    if not distances:
        return None
    diameter = 0.0
    for row in distances:
        for distance in row:
            if math.isfinite(distance):
                diameter = max(diameter, float(distance))
    return float(diameter)


def compute_avg_shortest_path_length(graph: GraphInput) -> float | None:
    """Compute average shortest path length of the largest connected component.

    Parameters
    ----------
    graph
        Graph to analyze (directed or undirected).

    Returns
    -------
    float | None
        Average shortest path length, or None if empty.

    Examples
    --------
    >>> g = RxGraphStore.undirected()
    >>> g.add_weighted_edge(0, 1, weight=1.0)
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> round(compute_avg_shortest_path_length(g), 2)
    1.33
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return None
    work_store = to_undirected_store(store)
    largest = _largest_component(work_store)
    if largest is None:
        return None
    if len(largest) <= 1:
        return 0.0
    undirected_graph = _undirected_graph(work_store)
    ordered_indices = [
        work_store.id_to_index[node_id]
        for node_id in largest
        if node_id in work_store.id_to_index
    ]
    ordered_indices.sort(key=lambda idx: stable_key(work_store.index_to_id[idx]))
    subgraph, _node_map = undirected_graph.subgraph_with_nodemap(
        ordered_indices,
        preserve_attrs=True,
    )
    return graph_unweighted_average_shortest_path_length(subgraph)


def compute_condensation_layer_count(graph: GraphInput) -> int | None:
    """Compute the number of layers in the SCC condensation DAG.

    The condensation DAG collapses each strongly connected component
    into a single node. This function returns the number of topological
    layers (the longest path length + 1) in that condensation DAG.

    Parameters
    ----------
    graph
        Directed graph to analyze.

    Returns
    -------
    int | None
        Number of layers, or None if the graph is empty or undirected.

    Examples
    --------
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> g.add_weighted_edge(3, 4, weight=1.0)
    >>> compute_condensation_layer_count(g)
    4
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0 or not store.is_directed:
        return None
    condensed, _membership = condensation_store(store)
    generations = topological_generations_by_id(condensed)
    return len(generations)


def compute_graph_statistics(graph: GraphInput) -> GraphStatistics:
    """Compute summary statistics for a directed graph.

    Parameters
    ----------
    graph
        Directed graph to analyze.

    Returns
    -------
    GraphStatistics
        Summary statistics for the graph.

    Examples
    --------
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge("a", "b", weight=1.0)
    >>> g.add_weighted_edge("b", "c", weight=1.0)
    >>> stats = compute_graph_statistics(g)
    >>> stats.node_count
    3
    >>> stats.edge_count
    2
    >>> stats.is_dag
    True
    """
    store = ensure_store(graph)
    node_count = store.graph.num_nodes()
    edge_count = store.graph.num_edges()

    if node_count == 0:
        return GraphStatistics(
            node_count=0,
            edge_count=0,
            density=0.0,
            avg_in_degree=0.0,
            avg_out_degree=0.0,
            strongly_connected_components=0,
            weakly_connected_components=0,
            is_dag=True,
        )

    density = edge_count / (node_count * (node_count - 1))

    in_degrees = get_in_degree_values(store)
    out_degrees = get_out_degree_values(store)
    avg_in_degree = sum(in_degrees) / node_count if node_count else 0.0
    avg_out_degree = sum(out_degrees) / node_count if node_count else 0.0

    if store.is_directed:
        strongly_connected = len(strongly_connected_components_by_id(store))
        weakly_connected = len(weakly_connected_components_by_id(store))
        is_dag = is_directed_acyclic(store)
    else:
        strongly_connected = len(connected_components_by_id(store))
        weakly_connected = strongly_connected
        is_dag = True

    return GraphStatistics(
        node_count=node_count,
        edge_count=edge_count,
        density=density,
        avg_in_degree=avg_in_degree,
        avg_out_degree=avg_out_degree,
        strongly_connected_components=strongly_connected,
        weakly_connected_components=weakly_connected,
        is_dag=is_dag,
    )


__all__ = [
    "GraphStatistics",
    "compute_avg_shortest_path_length",
    "compute_condensation_layer_count",
    "compute_diameter_estimate",
    "compute_graph_statistics",
    "get_degree_values",
    "get_degrees",
    "get_in_degree_values",
    "get_in_degrees",
    "get_out_degree_values",
    "get_out_degrees",
]
