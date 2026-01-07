"""Pure computation for path-related graph metrics.

This module provides functions to compute path metrics for graphs,
including path counting, shortest path lengths, and reachability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import rustworkx as rx

from codeintel.build.graphs.rx.algos import GraphInput, ensure_directed_store

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable


def count_simple_paths(
    graph: GraphInput,
    sources: Iterable[Hashable],
    targets: Iterable[Hashable],
    *,
    max_paths: int,
    cutoff: int,
) -> int:
    """Count simple paths between source and target sets with hard limits.

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

    Examples
    --------
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([(1, 2), (2, 3), (1, 3)])
    >>> count_simple_paths(g, [1], [3], max_paths=10, cutoff=5)
    2
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0
    directed_graph = cast("rx.PyDiGraph", store.graph)
    count = 0
    for source in sources:
        for target in targets:
            if count >= max_paths:
                return count
            source_idx = store.id_to_index.get(source)
            target_idx = store.id_to_index.get(target)
            if source_idx is None or target_idx is None:
                continue
            try:
                paths = rx.digraph_all_simple_paths(
                    directed_graph,
                    source_idx,
                    target_idx,
                    cutoff=cutoff,
                )
                for _ in paths:
                    count += 1
                    if count >= max_paths:
                        return count
            except (rx.InvalidNode, rx.NoPathFound, rx.NullGraph):
                continue
    return count


def compute_avg_shortest_path_from_source(
    graph: GraphInput,
    source: Hashable,
) -> float:
    """Compute average shortest path length from a single source.

    Parameters
    ----------
    graph
        Directed graph to analyze.
    source
        Source node for path computation.

    Returns
    -------
    float
        Average shortest path length from source, or 0.0 if no paths exist.

    Examples
    --------
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([(1, 2), (2, 3)])
    >>> round(compute_avg_shortest_path_from_source(g, 1), 2)
    1.0
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0.0
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return 0.0
    directed_graph = cast("rx.PyDiGraph", store.graph)
    try:
        lengths = rx.digraph_dijkstra_shortest_path_lengths(
            directed_graph,
            source_idx,
            lambda _payload: 1.0,
        )
    except (rx.InvalidNode, rx.NullGraph):
        return 0.0
    return sum(lengths.values()) / len(lengths) if lengths else 0.0


def compute_reachable_nodes(
    graph: GraphInput,
    source: Hashable,
) -> set[Hashable]:
    """Compute set of nodes reachable from source (including source).

    Parameters
    ----------
    graph
        Directed graph to analyze.
    source
        Source node for reachability computation.

    Returns
    -------
    set[Hashable]
        Reachable node identifiers including the source.

    Examples
    --------
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([(1, 2), (2, 3), (4, 5)])
    >>> sorted(compute_reachable_nodes(g, 1))
    [1, 2, 3]
    """
    store = ensure_directed_store(graph)
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return {source}
    directed_graph = cast("rx.PyDiGraph", store.graph)
    try:
        descendants = rx.descendants(directed_graph, source_idx)
    except (rx.InvalidNode, rx.NullGraph):
        descendants = set()
    nodes = {store.index_to_id[idx] for idx in descendants}
    nodes.add(source)
    return nodes


__all__ = [
    "compute_avg_shortest_path_from_source",
    "compute_reachable_nodes",
    "count_simple_paths",
]
