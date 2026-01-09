"""Pure computation for path-related graph metrics.

This module provides functions to compute path metrics for graphs,
including path counting, shortest path lengths, and reachability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import rustworkx as rx

from codeintel.build.graphs.rx.algos import (
    GraphAlgoConfig,
    GraphInput,
    constant_weight_fn,
    edge_cost_weight_fn,
    ensure_directed_store,
    resolve_weight_context,
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
    directed_graph = cast("rx.PyDiGraph", store.graph)
    weight_fn = constant_weight_fn()
    if weight is not None:
        weight_ctx = resolve_weight_context(
            store,
            algo_config=algo_config,
            nan_policy=nan_policy,
        )
        weight_fn = edge_cost_weight_fn(context=weight_ctx)
    try:
        lengths = rx.digraph_dijkstra_shortest_path_lengths(
            directed_graph,
            source_idx,
            weight_fn,
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
    >>> from codeintel.build.graphs.rx.store import RxGraphStore
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> g.add_weighted_edge(4, 5, weight=1.0)
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
