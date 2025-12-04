"""Pure control flow graph metric computation functions.

This module provides stateless functions for computing CFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

import networkx as nx

# Type variable for node types in CFG computations
NodeT = TypeVar("NodeT", bound=Hashable)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DominanceMetrics:
    """Dominance metrics for control flow graphs.

    Attributes
    ----------
    depth
        Depth in dominator tree (root = 0).
    frontier_size
        Size of dominance frontier.
    is_loop_header
        Whether node is a natural loop header.
    """

    depth: int
    frontier_size: int
    is_loop_header: bool


def compute_dominator_tree(
    graph: nx.DiGraph,
    entry: Hashable,
) -> dict[Hashable, Hashable | None]:
    """Compute immediate dominators for all nodes.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry
        Entry node (typically function start).

    Returns
    -------
    dict[Hashable, Hashable | None]
        Node to immediate dominator mapping. Entry node maps to None.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if entry not in graph:
        return {}

    try:
        idoms = nx.immediate_dominators(graph, entry)
    except nx.NetworkXError as exc:
        log.warning("Dominator computation failed: %s", exc)
        return {}

    # Entry dominates itself in NetworkX output; we represent as None
    result: dict[Hashable, Hashable | None] = {}
    for node, idom in idoms.items():
        result[node] = None if node == entry else idom
    return result


def compute_dominance_frontier(
    graph: nx.DiGraph,
    entry: Hashable,
) -> dict[Hashable, frozenset[Hashable]]:
    """Compute dominance frontier for all nodes.

    The dominance frontier of node n is the set of nodes where n's
    dominance ends.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry
        Entry node.

    Returns
    -------
    dict[Hashable, frozenset[Hashable]]
        Node to dominance frontier mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if entry not in graph:
        return {}

    try:
        frontiers = nx.dominance_frontiers(graph, entry)
    except nx.NetworkXError as exc:
        log.warning("Dominance frontier computation failed: %s", exc)
        return {}

    return {node: frozenset(frontier) for node, frontier in frontiers.items()}


def compute_dominator_depths(
    idoms: dict[NodeT, NodeT | None],
) -> dict[NodeT, int]:
    """Compute depth in dominator tree for all nodes.

    Parameters
    ----------
    idoms
        Immediate dominator mapping from compute_dominator_tree.

    Returns
    -------
    dict[NodeT, int]
        Node to depth mapping (root = 0).
    """
    if not idoms:
        return {}

    depths: dict[NodeT, int] = {}

    def get_depth(node: NodeT) -> int:
        if node in depths:
            return depths[node]
        idom = idoms.get(node)
        if idom is None:
            depths[node] = 0
        else:
            depths[node] = get_depth(idom) + 1
        return depths[node]

    for node in idoms:
        get_depth(node)

    return depths


def find_natural_loop_headers(
    graph: nx.DiGraph,
    entry: Hashable,
) -> set[Hashable]:
    """Find natural loop headers in a control flow graph.

    A natural loop header is a node that dominates a predecessor
    (forming a back edge).

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry
        Entry node.

    Returns
    -------
    set[Hashable]
        Set of loop header nodes.
    """
    if graph.number_of_nodes() == 0:
        return set()

    if entry not in graph:
        return set()

    try:
        idoms = nx.immediate_dominators(graph, entry)
    except nx.NetworkXError:
        return set()

    # Build dominance relation (transitive closure)
    dominates: dict[Hashable, set[Hashable]] = {node: set() for node in graph.nodes()}
    for node in graph.nodes():
        current: Hashable | None = node
        while current is not None:
            dominates[current].add(node)
            current = idoms.get(current)
            if current == node:  # Prevent infinite loop at entry
                break

    # Find back edges (edge n -> h where h dominates n)
    headers: set[Hashable] = set()
    for node in graph.nodes():
        for succ in graph.successors(node):
            if node in dominates.get(succ, set()):
                headers.add(succ)

    return headers


def compute_cfg_longest_path(
    graph: nx.DiGraph,
) -> int:
    """Compute longest path length in a CFG.

    For cyclic graphs, this computes on the DAG after condensation.

    Parameters
    ----------
    graph
        Control flow graph (directed).

    Returns
    -------
    int
        Longest path length (number of edges).
    """
    if graph.number_of_nodes() == 0:
        return 0

    # Check if acyclic
    if nx.is_directed_acyclic_graph(graph):
        return int(nx.dag_longest_path_length(graph))

    # For cyclic graphs, compute on condensation
    condensation = nx.condensation(graph)
    if condensation.number_of_nodes() == 0:
        return 0

    return int(nx.dag_longest_path_length(condensation))


def compute_all_dominance(
    graph: nx.DiGraph,
    entry: Hashable,
) -> dict[Hashable, DominanceMetrics]:
    """Compute all dominance-related metrics for CFG nodes.

    Parameters
    ----------
    graph
        Control flow graph.
    entry
        Entry node.

    Returns
    -------
    dict[Hashable, DominanceMetrics]
        Node to dominance metrics mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    idoms = compute_dominator_tree(graph, entry)
    frontiers = compute_dominance_frontier(graph, entry)
    depths = compute_dominator_depths(idoms)
    loop_headers = find_natural_loop_headers(graph, entry)

    return {
        node: DominanceMetrics(
            depth=depths.get(node, 0),
            frontier_size=len(frontiers.get(node, frozenset())),
            is_loop_header=node in loop_headers,
        )
        for node in graph.nodes()
    }


__all__ = [
    "DominanceMetrics",
    "compute_all_dominance",
    "compute_cfg_longest_path",
    "compute_dominance_frontier",
    "compute_dominator_depths",
    "compute_dominator_tree",
    "find_natural_loop_headers",
]
