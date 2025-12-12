"""Pure data flow graph metric computation functions.

This module provides stateless functions for computing DFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

from codeintel.graphs.compute.metrics.components import (
    find_strongly_connected,
    find_weakly_connected,
)

if TYPE_CHECKING:
    from codeintel.graphs.compute.metrics.components import (
        ComponentInfo,
    )

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


__all__ = [
    "DFGPathStats",
    "compute_def_use_chains",
    "compute_dfg_components",
    "compute_dfg_density",
    "compute_dfg_path_lengths",
    "compute_use_def_chains",
    "find_dfg_cycles",
]
