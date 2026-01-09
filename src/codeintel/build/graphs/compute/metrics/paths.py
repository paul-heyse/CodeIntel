"""Pure computation for path-related graph metrics.

This module provides functions to compute path metrics for graphs,
including path counting, shortest path lengths, and reachability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.graphs.rx.algos import (
    GraphAlgoConfig,
    GraphInput,
    descendants_by_id,
    digraph_shortest_path_lengths_by_id,
    ensure_directed_store,
    simple_paths_by_id,
)
from codeintel.build.graphs.rx.normalize import NanPolicy

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
    >>> from codeintel.build.graphs.rx.store import RxGraphStore
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> g.add_weighted_edge(1, 3, weight=1.0)
    >>> count_simple_paths(g, [1], [3], max_paths=10, cutoff=5)
    2
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0
    count = 0
    for source in sources:
        for target in targets:
            if count >= max_paths:
                return count
            paths = simple_paths_by_id(
                store,
                source,
                target,
                cutoff=cutoff,
                limit=max_paths - count,
            )
            count += len(paths)
    return count


def compute_avg_shortest_path_from_source(
    graph: GraphInput,
    source: Hashable,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> float:
    """Compute average shortest path length from a single source.

    Parameters
    ----------
    graph
        Directed graph to analyze.
    source
        Source node for path computation.
    weight
        Optional edge weight attribute for weighted shortest paths.
    nan_policy
        Optional NaN handling policy for weights.
    algo_config
        Optional algorithm configuration for weight semantics.

    Returns
    -------
    float
        Average shortest path length from source, or 0.0 if no paths exist.

    Examples
    --------
    >>> from codeintel.build.graphs.rx.store import RxGraphStore
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> round(compute_avg_shortest_path_from_source(g, 1), 2)
    1.0
    """
    store = ensure_directed_store(graph, weight=weight, nan_policy=nan_policy)
    if store.graph.num_nodes() == 0:
        return 0.0
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return 0.0
    lengths = digraph_shortest_path_lengths_by_id(
        store,
        source,
        weight=weight,
        nan_policy=nan_policy,
        algo_config=algo_config,
    )
    if not lengths:
        return 0.0
    return sum(lengths.values()) / len(lengths)


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
    >>> from codeintel.build.graphs.rx.store import RxGraphStore
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> g.add_weighted_edge(4, 5, weight=1.0)
    >>> sorted(compute_reachable_nodes(g, 1))
    [1, 2, 3]
    """
    store = ensure_directed_store(graph)
    return descendants_by_id(store, source, include_source=True)


__all__ = [
    "compute_avg_shortest_path_from_source",
    "compute_reachable_nodes",
    "count_simple_paths",
]
