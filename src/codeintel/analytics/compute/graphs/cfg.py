"""Control flow graph (CFG) metric computation.

This module provides functions for computing metrics on control flow graphs,
including dominator tree analysis and CFG-specific centrality metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.exception import NetworkXError

from codeintel.analytics.compute.graphs.centrality import centrality_directed
from codeintel.analytics.compute.graphs.types import DominanceMetrics
from codeintel.graphs.compute.metrics.cfg import (
    compute_cfg_longest_path,
    compute_dominance_frontier,
    compute_dominator_depths,
    compute_dominator_tree,
)
from codeintel.graphs.compute.metrics.paths import (
    compute_avg_shortest_path_from_source,
    compute_reachable_nodes,
)

if TYPE_CHECKING:
    from codeintel.analytics.compute.graphs.types import CentralityBundle
    from codeintel.analytics.runtime.context import GraphContext


def cfg_dominance_metrics(graph: nx.DiGraph, entry_idx: int) -> DominanceMetrics:
    """Compute dominator tree depth and frontier sizes for a CFG.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry_idx
        Entry block index.

    Returns
    -------
    DominanceMetrics
        Dominator depth, frontier sizes, and tree height.
    """
    idoms = compute_dominator_tree(graph, entry_idx)
    dom_depth = compute_dominator_depths(idoms)
    frontiers = compute_dominance_frontier(graph, entry_idx)
    frontier_sizes = {node: len(frontiers.get(node, frozenset())) for node in graph.nodes}
    height = max(dom_depth.values()) if dom_depth else None

    return DominanceMetrics(depth=dom_depth, frontier_sizes=frontier_sizes, tree_height=height)


def cfg_centralities(
    graph: nx.DiGraph,
    entry_idx: int,
    *,
    ctx: GraphContext,
) -> tuple[CentralityBundle, DominanceMetrics]:
    """Compute CFG centralities and dominance metrics.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry_idx
        Entry block index.
    ctx
        Graph context for computation parameters.

    Returns
    -------
    tuple[CentralityBundle, DominanceMetrics]
        Centralities and dominance metadata.
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

    When the graph is a DAG the search is limited to reachable nodes; otherwise
    the condensation DAG is used.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry_idx
        Entry block index.
    is_dag
        Whether the graph is a DAG (computed if None).

    Returns
    -------
    int
        Length of the longest path reachable from the entry block.
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

    Delegates to graphs.compute.metrics.paths.compute_avg_shortest_path_from_source.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry_idx
        Entry block index.

    Returns
    -------
    float
        Average shortest path length.
    """
    return compute_avg_shortest_path_from_source(graph, entry_idx)


def cfg_reachable_nodes(graph: nx.DiGraph, entry_idx: int) -> set[Any]:
    """Return the set of nodes reachable from the entry node.

    Delegates to graphs.compute.metrics.paths.compute_reachable_nodes.

    Parameters
    ----------
    graph
        Control flow graph (directed).
    entry_idx
        Entry block index.

    Returns
    -------
    set[Any]
        Reachable node identifiers including the entry.
    """
    return compute_reachable_nodes(graph, entry_idx)


def build_cfg_graph(
    blocks: list[tuple[int, str, int, int]],
    edges: list[tuple[int, int, str]],
) -> tuple[nx.DiGraph, int, int]:
    """Build a control-flow graph from block and edge tuples.

    Parameters
    ----------
    blocks
        List of (idx, kind, in_degree, out_degree) tuples.
    edges
        List of (src, dst, edge_type) tuples.

    Returns
    -------
    tuple[nx.DiGraph, int, int]
        CFG, entry block index, and exit block index.
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
        entry_idx = min(graph.nodes)
    if exit_idx is None:
        exits = [n for n, deg in out_deg_map.items() if deg == 0]
        exit_idx = exits[0] if exits else (entry_idx if entry_idx is not None else 0)
    return graph, entry_idx or 0, exit_idx or 0


__all__ = [
    "build_cfg_graph",
    "cfg_avg_shortest_path_length",
    "cfg_centralities",
    "cfg_dominance_metrics",
    "cfg_longest_path_length",
    "cfg_reachable_nodes",
]
