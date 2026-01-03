"""Pure control flow graph metric computation functions.

This module provides stateless functions for computing CFG-specific
metrics without any database or file I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import networkx as nx
from networkx.exception import NetworkXError

from codeintel.build.graphs.compute.metrics.centrality import centrality_directed
from codeintel.build.graphs.compute.metrics.paths import (
    compute_avg_shortest_path_from_source,
    compute_reachable_nodes,
)
from codeintel.build.graphs.compute.metrics.types import (
    CentralityBundle,
)
from codeintel.build.graphs.compute.metrics.types import (
    DominanceMetrics as DominanceSummary,
)

if TYPE_CHECKING:
    from codeintel.build.graphs.runtime.context import GraphContext

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
    except NetworkXError as exc:
        log.warning("Dominator computation failed: %s", exc)
        return {}

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
    except NetworkXError as exc:
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
    except NetworkXError:
        return set()

    dominates: dict[Hashable, set[Hashable]] = {node: set() for node in graph.nodes()}
    for node in graph.nodes():
        current: Hashable | None = node
        while current is not None:
            dominates[current].add(node)
            current = idoms.get(current)
            if current == node:
                break

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

    if nx.is_directed_acyclic_graph(graph):
        return int(nx.dag_longest_path_length(graph))

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


def cfg_dominance_metrics(graph: nx.DiGraph, entry_idx: int) -> DominanceSummary:
    """Compute dominator tree depth and frontier sizes for a CFG.

    Returns
    -------
    DominanceSummary
        Dominance depth and frontier sizes for CFG nodes.
    """
    idoms = compute_dominator_tree(graph, entry_idx)
    dom_depth = compute_dominator_depths(idoms)
    frontiers = compute_dominance_frontier(graph, entry_idx)
    frontier_sizes = {node: len(frontiers.get(node, frozenset())) for node in graph.nodes}
    height = max(dom_depth.values()) if dom_depth else None

    return DominanceSummary(
        depth=dom_depth,
        frontier_sizes=frontier_sizes,
        tree_height=height,
    )


def cfg_centralities(
    graph: nx.DiGraph,
    entry_idx: int,
    *,
    ctx: GraphContext,
) -> tuple[CentralityBundle, DominanceSummary]:
    """Compute CFG centralities and dominance metrics.

    Returns
    -------
    tuple[CentralityBundle, DominanceSummary]
        Centrality bundle and dominance summary for the CFG.
    """
    dominance = cfg_dominance_metrics(graph, entry_idx)
    centrality = centrality_directed(
        graph,
        ctx,
        weight=None,
        include_eigen=True,
    )
    return centrality, dominance


def cfg_longest_path_length(
    graph: nx.DiGraph,
    entry_idx: int,
    *,
    is_dag: bool | None = None,
) -> int:
    """Compute the longest path length for a CFG.

    Returns
    -------
    int
        Longest path length from the entry node.
    """
    if graph.number_of_nodes() == 0:
        return 0

    if is_dag is None:
        is_dag = nx.is_directed_acyclic_graph(graph)

    if is_dag:
        try:
            reachable = nx.descendants(graph, entry_idx) | {entry_idx}
            subgraph = graph.subgraph(reachable).copy()
        except NetworkXError:
            return 0
        return compute_cfg_longest_path(nx.DiGraph(subgraph))

    return compute_cfg_longest_path(graph)


def cfg_avg_shortest_path_length(graph: nx.DiGraph, entry_idx: int) -> float:
    """Return the average shortest path length from the entry block.

    Returns
    -------
    float
        Average shortest path length from the entry node.
    """
    return compute_avg_shortest_path_from_source(graph, entry_idx)


def cfg_reachable_nodes(graph: nx.DiGraph, entry_idx: int) -> set[Any]:
    """Return the set of nodes reachable from the entry node.

    Returns
    -------
    set[Any]
        Reachable nodes in the CFG.
    """
    return set(compute_reachable_nodes(graph, entry_idx))


def build_cfg_graph(
    blocks: list[tuple[int, str, int, int]],
    edges: list[tuple[int, int, str]],
) -> tuple[nx.DiGraph, int, int]:
    """Build a control-flow graph from block and edge tuples.

    Returns
    -------
    tuple[nx.DiGraph, int, int]
        Graph, entry node id, and exit node id.
    """
    graph = nx.DiGraph()
    entry_idx = None
    exit_idx = None
    out_deg_map: dict[int, int] = {}
    for idx, kind, in_deg, out_deg in blocks:
        graph.add_node(idx, kind=kind, in_degree=in_deg, out_degree=out_deg)
        if kind == "entry":
            entry_idx = idx
        if kind == "exit":
            exit_idx = idx
        out_deg_map[idx] = out_deg
    for src, dst, edge_type in edges:
        graph.add_edge(src, dst, edge_type=edge_type)
    if entry_idx is None and graph.nodes:
        entry_idx = min(int(str(node)) for node in graph.nodes)
    if exit_idx is None:
        exits = [node for node, deg in out_deg_map.items() if deg == 0]
        exit_idx = exits[0] if exits else (entry_idx if entry_idx is not None else 0)
    return graph, entry_idx or 0, exit_idx or 0


__all__ = [
    "DominanceMetrics",
    "build_cfg_graph",
    "cfg_avg_shortest_path_length",
    "cfg_centralities",
    "cfg_dominance_metrics",
    "cfg_longest_path_length",
    "cfg_reachable_nodes",
    "compute_all_dominance",
    "compute_cfg_longest_path",
    "compute_dominance_frontier",
    "compute_dominator_depths",
    "compute_dominator_tree",
    "find_natural_loop_headers",
]
