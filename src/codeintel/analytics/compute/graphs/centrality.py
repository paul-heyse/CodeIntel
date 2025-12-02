"""Pure computation for graph centrality metrics.

This module provides functions to compute centrality metrics
on networkx graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class CentralityMetrics:
    """Centrality metrics for a node.

    Attributes
    ----------
    node_id
        Identifier for the node.
    pagerank
        PageRank score.
    betweenness
        Betweenness centrality.
    in_degree
        In-degree count.
    out_degree
        Out-degree count.
    """

    node_id: str
    pagerank: float
    betweenness: float
    in_degree: int
    out_degree: int


def compute_pagerank(
    graph: nx.DiGraph,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute PageRank for all nodes in a directed graph.

    Parameters
    ----------
    graph
        Directed graph to analyze.
    alpha
        Damping parameter (default: 0.85).
    max_iter
        Maximum iterations (default: 100).
    tol
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    dict[str, float]
        Mapping of node IDs to PageRank scores.

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    >>> pr = compute_pagerank(g)
    >>> len(pr)
    3
    """
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}

    try:
        result = nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol)
        return {str(k): float(v) for k, v in result.items()}
    except nx.PowerIterationFailedConvergence:
        # Fall back to simpler scoring
        return {str(n): 1.0 / graph.number_of_nodes() for n in graph.nodes()}


def compute_betweenness(
    graph: nx.DiGraph,
    *,
    normalized: bool = True,
    k: int | None = None,
) -> dict[str, float]:
    """Compute betweenness centrality for all nodes.

    Parameters
    ----------
    graph
        Directed graph to analyze.
    normalized
        Whether to normalize values (default: True).
    k
        Sample size for approximate computation (default: None for exact).

    Returns
    -------
    dict[str, float]
        Mapping of node IDs to betweenness scores.

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.DiGraph()
    >>> g.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    >>> bc = compute_betweenness(g)
    >>> len(bc)
    3
    """
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {}

    result = nx.betweenness_centrality(graph, normalized=normalized, k=k)
    return {str(k): float(v) for k, v in result.items()}


__all__ = [
    "CentralityMetrics",
    "compute_betweenness",
    "compute_pagerank",
]

