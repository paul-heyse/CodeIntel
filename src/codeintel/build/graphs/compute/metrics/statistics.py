"""Pure computation for graph statistics.

This module provides functions to compute summary statistics
for rustworkx graph stores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import rustworkx as rx

from codeintel.build.graphs.rx.algos import (
    GraphInput,
    ensure_directed_store,
    ensure_store,
    to_undirected_store,
)
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


def _directed_graph(store: RxGraphStore) -> rx.PyDiGraph:
    if not store.is_directed:
        message = "Expected a directed graph store"
        raise ValueError(message)
    return cast("rx.PyDiGraph", store.graph)


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
    store = ensure_directed_store(graph)
    directed_graph = _directed_graph(store)
    return [
        (node_id, directed_graph.in_degree(store.id_to_index[node_id]))
        for node_id in store.node_ids()
    ]


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
    store = ensure_directed_store(graph)
    directed_graph = _directed_graph(store)
    return [
        (node_id, directed_graph.out_degree(store.id_to_index[node_id]))
        for node_id in store.node_ids()
    ]


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
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    undirected_graph = _undirected_graph(work_store)
    return [
        (node_id, undirected_graph.degree(work_store.id_to_index[node_id]))
        for node_id in work_store.node_ids()
    ]


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


def _component_sort_key(store: RxGraphStore, component: set[int]) -> tuple[int, tuple[str, str]]:
    if not component:
        return (0, ("", ""))
    smallest = min(
        (store.index_to_id[idx] for idx in component),
        key=stable_key,
    )
    return (len(component), stable_key(smallest))


def _largest_component(store: RxGraphStore) -> set[int] | None:
    undirected_graph = _undirected_graph(store)
    components = [set(comp) for comp in rx.connected_components(undirected_graph)]
    if not components:
        return None
    return max(components, key=lambda comp: _component_sort_key(store, comp))


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
    subgraph = undirected_graph.subgraph(list(largest), preserve_attrs=True)
    try:
        lengths = rx.graph_all_pairs_dijkstra_path_lengths(subgraph, lambda _payload: 1.0)
    except rx.NullGraph:
        return None
    diameter = 0.0
    for targets in lengths.values():
        if targets:
            diameter = max(diameter, max(targets.values(), default=0))
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
    subgraph = undirected_graph.subgraph(list(largest), preserve_attrs=True)
    try:
        lengths = rx.graph_all_pairs_dijkstra_path_lengths(subgraph, lambda _payload: 1.0)
    except rx.NullGraph:
        return None
    total = 0.0
    count = 0
    for targets in lengths.values():
        total += sum(targets.values())
        count += len(targets)
    if count == 0:
        return 0.0
    return total / count


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
    condensed = _condensation_graph(store)
    return _layer_count(condensed)


def _condensation_graph(store: RxGraphStore) -> rx.PyDiGraph:
    directed_graph = _directed_graph(store)
    sccs = [set(comp) for comp in rx.strongly_connected_components(directed_graph)]
    if not sccs:
        return rx.PyDiGraph(multigraph=False)
    sorted_sccs = sorted(sccs, key=lambda comp: _component_sort_key(store, comp))
    comp_map = _component_membership(sorted_sccs)
    condensed = rx.PyDiGraph(multigraph=False)
    condensed.add_nodes_from(range(len(sorted_sccs)))
    for src_idx, dst_idx in store.graph.edge_list():
        src_comp = comp_map.get(src_idx)
        dst_comp = comp_map.get(dst_idx)
        if src_comp is None or dst_comp is None or src_comp == dst_comp:
            continue
        condensed.add_edge(src_comp, dst_comp, 1)
    return condensed


def _component_membership(components: Sequence[set[int]]) -> dict[int, int]:
    comp_map: dict[int, int] = {}
    for comp_id, comp in enumerate(components):
        for node_idx in comp:
            comp_map[node_idx] = comp_id
    return comp_map


def _layer_count(graph: rx.PyDiGraph) -> int:
    if graph.num_nodes() == 0:
        return 0
    layers: dict[int, int] = {
        node_idx: 0 for node_idx in graph.node_indices() if graph.in_degree(node_idx) == 0
    }
    for node_idx in rx.topological_sort(graph):
        base = layers.get(node_idx, 0)
        for succ in graph.successor_indices(node_idx):
            layers[succ] = max(layers.get(succ, 0), base + 1)
    return max(layers.values(), default=0) + 1


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
        directed_graph = _directed_graph(store)
        strongly_connected = len(list(rx.strongly_connected_components(directed_graph)))
        weakly_connected = len(list(rx.weakly_connected_components(directed_graph)))
        is_dag = rx.is_directed_acyclic_graph(directed_graph)
    else:
        undirected_graph = _undirected_graph(store)
        strongly_connected = len(list(rx.connected_components(undirected_graph)))
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
