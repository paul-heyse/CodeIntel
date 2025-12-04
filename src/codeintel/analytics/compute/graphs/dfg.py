"""Data flow graph (DFG) metric computation.

This module provides functions for computing metrics on data flow graphs,
including component analysis, path analysis, and centrality metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.exception import NetworkXError

from codeintel.analytics.compute.graphs.centrality import centrality_directed
from codeintel.graphs.compute.metrics.dfg import compute_dfg_components

if TYPE_CHECKING:
    from codeintel.analytics.graphs.runtime import GraphContext


def dfg_component_stats(graph: nx.DiGraph) -> tuple[int, list[set[int]], bool]:
    """Return connected component stats for DFG graphs.

    Parameters
    ----------
    graph
        Data flow graph (directed).

    Returns
    -------
    tuple[int, list[set[int]], bool]
        Component count, components, and cycle flag.
    """
    sccs, wccs = compute_dfg_components(graph)
    components: list[set[int]] = [set(wcc.nodes) for wcc in wccs]
    has_cycles = any(scc.size > 1 for scc in sccs)
    return len(components), components, has_cycles


def dfg_path_lengths(graph: nx.DiGraph) -> tuple[int, float]:
    """Return longest path length and average shortest path length for DFGs.

    Parameters
    ----------
    graph
        Data flow graph (directed).

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

    Parameters
    ----------
    graph
        Data flow graph (directed).
    ctx
        Graph context for computation parameters.

    Returns
    -------
    tuple[dict[Any, float], dict[Any, float]]
        Betweenness and eigenvector centrality.
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

    Parameters
    ----------
    edges
        List of (src, dst, src_symbol, dst_symbol, via_phi, use_kind) tuples.

    Returns
    -------
    tuple[nx.DiGraph, int, int]
        Graph, phi edge count, and distinct symbol count.
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
    "build_dfg_graph",
    "dfg_centralities",
    "dfg_component_stats",
    "dfg_path_lengths",
]
