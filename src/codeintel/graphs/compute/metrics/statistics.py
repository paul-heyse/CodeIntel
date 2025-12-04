"""Pure computation for graph statistics.

This module provides functions to compute summary statistics
for networkx graphs.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import networkx as nx

# Type variable for node types (typically int for GOIDs)
NodeT = TypeVar("NodeT", bound=Hashable)

# NetworkX DegreeView is Iterable[tuple[node, degree]] but stubs type it as int
# Use Any for node type since graphs can have any hashable node type
DegreeViewT = Iterable[tuple[Any, int]]


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


def get_in_degrees(graph: nx.DiGraph) -> list[tuple[Any, int]]:
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
    >>> g = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    >>> get_in_degrees(g)
    [(1, 0), (2, 1), (3, 2)]
    """
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.in_degree())
    return [(node, degree) for node, degree in degrees]


def get_out_degrees(graph: nx.DiGraph) -> list[tuple[Any, int]]:
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
    >>> g = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    >>> get_out_degrees(g)
    [(1, 2), (2, 1), (3, 0)]
    """
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.out_degree())
    return [(node, degree) for node, degree in degrees]


def get_degrees(graph: nx.Graph) -> list[tuple[Any, int]]:
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
    >>> g = nx.Graph([(1, 2), (1, 3), (2, 3)])
    >>> get_degrees(g)
    [(1, 2), (2, 2), (3, 2)]
    """
    # NetworkX stubs have issues with degree(); use attribute access
    degrees = cast("DegreeViewT", graph.degree)
    return [(node, degree) for node, degree in degrees]


def get_in_degree_values(graph: nx.DiGraph) -> list[int]:
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
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.in_degree())
    return [degree for _, degree in degrees]


def get_out_degree_values(graph: nx.DiGraph) -> list[int]:
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
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.out_degree())
    return [degree for _, degree in degrees]


def get_degree_values(graph: nx.Graph) -> list[int]:
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
    # NetworkX stubs have issues with degree(); use attribute access
    degrees = cast("DegreeViewT", graph.degree)
    return [degree for _, degree in degrees]


def compute_diameter_estimate(graph: nx.Graph) -> float | None:
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
    >>> g = nx.path_graph(5)
    >>> compute_diameter_estimate(g)
    4.0
    """
    if graph.number_of_nodes() == 0:
        return None
    undirected = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
    components = list(nx.connected_components(undirected))
    if not components:
        return None
    largest = undirected.subgraph(max(components, key=len)).copy()
    try:
        return float(nx.approximation.diameter(largest))
    except nx.NetworkXError:
        return None


def compute_avg_shortest_path_length(graph: nx.Graph) -> float | None:
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
    >>> g = nx.path_graph(4)
    >>> round(compute_avg_shortest_path_length(g), 2)
    1.33
    """
    if graph.number_of_nodes() == 0:
        return None
    undirected = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph
    components = list(nx.connected_components(undirected))
    if not components:
        return None
    largest = undirected.subgraph(max(components, key=len)).copy()
    try:
        return float(nx.average_shortest_path_length(largest))
    except nx.NetworkXError:
        return None


def compute_condensation_layer_count(graph: nx.DiGraph) -> int | None:
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
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([(1, 2), (2, 3), (3, 4)])
    >>> compute_condensation_layer_count(g)
    4
    """
    if graph.number_of_nodes() == 0:
        return None
    condensation = nx.condensation(graph)
    if condensation.number_of_nodes() == 0:
        return 0
    # Compute topological layers for the condensation DAG
    layers: dict[int, int] = {
        node: 0 for node in condensation.nodes if condensation.in_degree(node) == 0
    }
    for node in nx.topological_sort(condensation):
        base = layers.get(node, 0)
        for succ in condensation.successors(node):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return max(layers.values(), default=0) + 1


def compute_graph_statistics(graph: nx.DiGraph) -> GraphStatistics:
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
    >>> import networkx as nx
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([("a", "b"), ("b", "c")])
    >>> stats = compute_graph_statistics(g)
    >>> stats.node_count
    3
    >>> stats.edge_count
    2
    >>> stats.is_dag
    True
    """
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

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

    # Density
    density = nx.density(graph)

    # Average degrees (using type-safe wrappers)
    in_degrees = get_in_degree_values(graph)
    out_degrees = get_out_degree_values(graph)
    avg_in_degree = sum(in_degrees) / node_count if node_count else 0.0
    avg_out_degree = sum(out_degrees) / node_count if node_count else 0.0

    # Connected components
    strongly_connected = nx.number_strongly_connected_components(graph)
    weakly_connected = nx.number_weakly_connected_components(graph)

    # DAG check
    is_dag = nx.is_directed_acyclic_graph(graph)

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
    "DegreeViewT",
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
