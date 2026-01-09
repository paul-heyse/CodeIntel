"""Pure data flow graph metric computation functions.

This module provides stateless functions for computing DFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.graphs.compute.metrics.centrality import centrality_directed
from codeintel.build.graphs.compute.metrics.components import (
    find_strongly_connected,
    find_weakly_connected,
)
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    bfs_distances_by_id,
    digraph_all_pairs_shortest_path_lengths_by_id,
    ensure_directed_store,
    predecessors_by_id,
    simple_cycles_by_id,
    successors_by_id,
)
from codeintel.build.graphs.rx.store import RxGraphStore

if TYPE_CHECKING:
    from codeintel.build.graphs.compute.metrics.components import (
        ComponentInfo,
    )
    from codeintel.build.graphs.runtime.context import GraphContext


@dataclass(frozen=True)
class DFGPathStats:
    """Path length statistics for a DFG node.

    Attributes
    ----------
    max_def_use_distance
        Maximum distance from definition to use.
    avg_def_use_distance
        Average distance from definition to uses.
    reach_count
        Number of nodes reachable from this node.
    """

    max_def_use_distance: int
    avg_def_use_distance: float
    reach_count: int


def compute_dfg_path_lengths(
    graph: GraphInput,
    *,
    max_depth: int = 100,
) -> dict[Any, DFGPathStats]:
    """Compute path length statistics for DFG nodes.

    Parameters
    ----------
    graph
        Data flow graph (directed).
    max_depth
        Maximum search depth (for performance).

    Returns
    -------
    dict[Any, DFGPathStats]
        Node to path statistics mapping.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}

    result: dict[Any, DFGPathStats] = {}
    for node_id in store.node_ids():
        distances = bfs_distances_by_id(store, node_id, max_depth=max_depth + 1)
        limit = max_depth + 1
        bounded = [
            distance
            for node, distance in distances.items()
            if node != node_id and 0 < distance <= limit
        ]
        if bounded:
            result[node_id] = DFGPathStats(
                max_def_use_distance=max(bounded),
                avg_def_use_distance=sum(bounded) / len(bounded),
                reach_count=len(bounded),
            )
            continue
        result[node_id] = DFGPathStats(
            max_def_use_distance=0,
            avg_def_use_distance=0.0,
            reach_count=0,
        )
    return result


def compute_dfg_components(
    graph: GraphInput,
) -> tuple[list[ComponentInfo], list[ComponentInfo]]:
    """Compute connected components for a DFG.

    Returns both strongly and weakly connected components.

    Parameters
    ----------
    graph
        Data flow graph (directed).

    Returns
    -------
    tuple[list[ComponentInfo], list[ComponentInfo]]
        (strongly_connected, weakly_connected) component lists.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return ([], [])

    scc_result = find_strongly_connected(graph)
    wccs = find_weakly_connected(graph)

    return (list(scc_result.components), wccs)


def compute_def_use_chains(
    graph: GraphInput,
) -> dict[Any, list[Any]]:
    """Compute def-use chains for each node.

    A def-use chain is the list of nodes that use a definition.

    Parameters
    ----------
    graph
        Data flow graph where edges represent def-use relationships.

    Returns
    -------
    dict[Any, list[Any]]
        Node to list of users mapping.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    result: dict[Any, list[Any]] = {}
    for node_id in store.node_ids():
        result[node_id] = successors_by_id(store, node_id)
    return result


def compute_use_def_chains(
    graph: GraphInput,
) -> dict[Any, list[Any]]:
    """Compute use-def chains for each node.

    A use-def chain is the list of definitions that reach a use.

    Parameters
    ----------
    graph
        Data flow graph where edges represent def-use relationships.

    Returns
    -------
    dict[Any, list[Any]]
        Node to list of definitions mapping.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    result: dict[Any, list[Any]] = {}
    for node_id in store.node_ids():
        result[node_id] = predecessors_by_id(store, node_id)
    return result


def compute_dfg_density(graph: GraphInput) -> float:
    """Compute edge density of a DFG.

    Parameters
    ----------
    graph
        Data flow graph.

    Returns
    -------
    float
        Edge density (0.0 to 1.0).
    """
    store = ensure_directed_store(graph)
    node_count = store.graph.num_nodes()
    if node_count <= 1:
        return 0.0
    max_edges = node_count * (node_count - 1)
    return store.graph.num_edges() / max_edges


def find_dfg_cycles(
    graph: GraphInput,
    *,
    limit: int = 100,
) -> list[list[Any]]:
    """Find cycles in a DFG (may indicate recursive data flow).

    Parameters
    ----------
    graph
        Data flow graph.
    limit
        Maximum number of cycles to return.

    Returns
    -------
    list[list[Any]]
        List of cycles as node sequences.
    """
    return simple_cycles_by_id(graph, limit=limit)


def dfg_component_stats(graph: GraphInput) -> tuple[int, list[set[Any]], bool]:
    """Return connected component stats for DFG graphs.

    Returns
    -------
    tuple[int, list[set[int]], bool]
        Component count, components, and whether cycles are present.
    """
    sccs, wccs = compute_dfg_components(graph)
    components: list[set[Any]] = [set(wcc.nodes) for wcc in wccs]
    has_cycles = any(scc.size > 1 for scc in sccs)
    return len(components), components, has_cycles


def dfg_path_lengths(graph: GraphInput) -> tuple[int, float]:
    """Return longest path length and average shortest path length for DFGs.

    Returns
    -------
    tuple[int, float]
        Longest path length and average shortest path length.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0, 0.0
    lengths = digraph_all_pairs_shortest_path_lengths_by_id(store)
    longest = 0.0
    total = 0.0
    count = 0
    for targets in lengths.values():
        if targets:
            longest = max(longest, float(max(targets.values(), default=0)))
            total += sum(targets.values())
            count += len(targets)
    avg = total / count if count else 0.0
    return int(longest), avg


def dfg_centralities(
    graph: GraphInput, ctx: GraphContext
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Compute DFG betweenness and eigenvector centralities.

    Returns
    -------
    tuple[dict[Any, float], dict[Any, float]]
        Betweenness and eigenvector centrality mappings.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}, {}
    centrality = centrality_directed(
        store,
        ctx,
        weight=None,
        include_eigen=True,
    )
    return centrality.betweenness, centrality.eigenvector


def build_dfg_graph(
    edges: list[tuple[int, int, str, str, bool, str]],
) -> tuple[RxGraphStore, int, int]:
    """Build a data-flow graph from edge tuples.

    Returns
    -------
    tuple[RxGraphStore, int, int]
        Graph, phi edge count, and symbol count.
    """
    graph = RxGraphStore.directed()
    phi_edges = 0
    symbols: set[str] = set()
    for src, dst, src_sym, dst_sym, via_phi, _use_kind in edges:
        graph.add_weighted_edge(src, dst, weight=1.0)
        symbols.add(src_sym)
        symbols.add(dst_sym)
        if via_phi:
            phi_edges += 1
    return graph, phi_edges, len(symbols)


__all__ = [
    "DFGPathStats",
    "build_dfg_graph",
    "compute_def_use_chains",
    "compute_dfg_components",
    "compute_dfg_density",
    "compute_dfg_path_lengths",
    "compute_use_def_chains",
    "dfg_centralities",
    "dfg_component_stats",
    "dfg_path_lengths",
    "find_dfg_cycles",
]
