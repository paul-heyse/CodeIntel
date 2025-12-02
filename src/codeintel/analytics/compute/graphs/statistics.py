"""Pure computation for graph statistics.

This module provides functions to compute summary statistics
for networkx graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from codeintel.analytics.compute.graphs.nx_types import (
    get_in_degree_values,
    get_out_degree_values,
)


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

    # Average degrees (using type-safe wrappers from nx_types)
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
    "GraphStatistics",
    "compute_graph_statistics",
]
