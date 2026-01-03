"""Pure data flow graph metric computation functions.

This module provides stateless functions for computing DFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.exception import NetworkXError

from codeintel.build.graphs.compute.metrics.centrality import centrality_directed
from codeintel.build.graphs.compute.metrics.components import (
    find_strongly_connected,
    find_weakly_connected,
)

if TYPE_CHECKING:
    from codeintel.build.graphs.compute.metrics.components import (
        ComponentInfo,
    )
    from codeintel.build.graphs.runtime.context import GraphContext

log = logging.getLogger(__name__)


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
    graph: nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    result: dict[Any, DFGPathStats] = {}

    for node in graph.nodes():
        distances: dict[Any, int] = {}
        queue = [(node, 0)]
        visited: set[Any] = {node}

        while queue:
            current, dist = queue.pop(0)
            if dist > max_depth:
                continue

            for succ in graph.successors(current):
                if succ not in visited:
                    visited.add(succ)
                    distances[succ] = dist + 1
                    queue.append((succ, dist + 1))

        if distances:
            path_lengths = list(distances.values())
            result[node] = DFGPathStats(
                max_def_use_distance=max(path_lengths),
                avg_def_use_distance=sum(path_lengths) / len(path_lengths),
                reach_count=len(distances),
            )
        else:
            result[node] = DFGPathStats(
                max_def_use_distance=0,
                avg_def_use_distance=0.0,
                reach_count=0,
            )

    return result


def compute_dfg_components(
    graph: nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return ([], [])

    scc_result = find_strongly_connected(graph)
    wccs = find_weakly_connected(graph)

    return (list(scc_result.components), wccs)


def compute_def_use_chains(
    graph: nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    return {node: list(graph.successors(node)) for node in graph.nodes()}


def compute_use_def_chains(
    graph: nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return {}

    return {node: list(graph.predecessors(node)) for node in graph.nodes()}


def compute_dfg_density(graph: nx.DiGraph) -> float:
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
    if graph.number_of_nodes() <= 1:
        return 0.0

    n = graph.number_of_nodes()
    max_edges = n * (n - 1)
    return graph.number_of_edges() / max_edges


def find_dfg_cycles(
    graph: nx.DiGraph,
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
    if graph.number_of_nodes() == 0:
        return []

    cycles: list[list[Any]] = []
    for cycle in nx.simple_cycles(graph):
        cycles.append(cycle)
        if len(cycles) >= limit:
            break

    return cycles


def dfg_component_stats(graph: nx.DiGraph) -> tuple[int, list[set[int]], bool]:
    """Return connected component stats for DFG graphs.

    Returns
    -------
    tuple[int, list[set[int]], bool]
        Component count, components, and whether cycles are present.
    """
    sccs, wccs = compute_dfg_components(graph)
    components: list[set[int]] = [set(wcc.nodes) for wcc in wccs]
    has_cycles = any(scc.size > 1 for scc in sccs)
    return len(components), components, has_cycles


def dfg_path_lengths(graph: nx.DiGraph) -> tuple[int, float]:
    """Return longest path length and average shortest path length for DFGs.

    Returns
    -------
    tuple[int, float]
        Longest path length and average shortest path length.
    """
    if graph.number_of_nodes() == 0:
        return 0, 0.0
    longest = 0
    try:
        lengths = dict(nx.all_pairs_shortest_path_length(graph))
    except NetworkXError:
        return 0, 0.0
    total = 0
    count = 0
    for targets in lengths.values():
        longest = max(longest, max(targets.values(), default=0))
        total += sum(targets.values())
        count += len(targets)
    avg = float(total) / count if count else 0.0
    return int(longest), avg


def dfg_centralities(
    graph: nx.DiGraph, ctx: GraphContext
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Compute DFG betweenness and eigenvector centralities.

    Returns
    -------
    tuple[dict[Any, float], dict[Any, float]]
        Betweenness and eigenvector centrality mappings.
    """
    if graph.number_of_nodes() == 0:
        return {}, {}
    centrality = centrality_directed(
        graph,
        ctx,
        weight=None,
        include_eigen=True,
    )
    return centrality.betweenness, centrality.eigenvector


def build_dfg_graph(
    edges: list[tuple[int, int, str, str, bool, str]],
) -> tuple[nx.DiGraph, int, int]:
    """Build a data-flow graph from edge tuples.

    Returns
    -------
    tuple[nx.DiGraph, int, int]
        Graph, phi edge count, and symbol count.
    """
    graph: nx.DiGraph = nx.DiGraph()
    phi_edges = 0
    symbols: set[str] = set()
    for src, dst, src_sym, dst_sym, via_phi, use_kind in edges:
        graph.add_edge(
            src,
            dst,
            src_symbol=src_sym,
            dst_symbol=dst_sym,
            via_phi=via_phi,
            use_kind=use_kind,
        )
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
