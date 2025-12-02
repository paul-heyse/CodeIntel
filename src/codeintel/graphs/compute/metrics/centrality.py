"""Pure centrality metric computation functions.

This module provides stateless functions for computing centrality metrics
on NetworkX graphs without any database or file I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import networkx as nx


@dataclass(frozen=True)
class CentralityMetrics:
    """Collection of centrality metrics for a node.

    Attributes
    ----------
    pagerank
        PageRank score.
    betweenness
        Betweenness centrality.
    closeness
        Closeness centrality.
    in_degree
        In-degree (for directed graphs).
    out_degree
        Out-degree (for directed graphs).
    degree
        Total degree.
    """

    pagerank: float
    betweenness: float
    closeness: float
    in_degree: int
    out_degree: int
    degree: int


def compute_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[Any, float]:
    """Compute PageRank for all nodes in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.
    alpha
        Damping factor.
    max_iter
        Maximum iterations.
    tol
        Convergence tolerance.

    Returns
    -------
    dict[N, float]
        Node to PageRank score mapping.

    Examples
    --------
    >>> g = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    >>> pr = compute_pagerank(g)
    >>> len(pr)
    3
    """
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        # Fall back to uniform distribution
        n = graph.number_of_nodes()
        return dict.fromkeys(graph.nodes(), 1.0 / n)


def compute_betweenness(
    graph: nx.Graph | nx.DiGraph,
    *,
    normalized: bool = True,
    k: int | None = None,
) -> dict[Any, float]:
    """Compute betweenness centrality for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    normalized
        Whether to normalize values.
    k
        Number of sample nodes (None for exact computation).

    Returns
    -------
    dict[N, float]
        Node to betweenness centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(graph, normalized=normalized, k=k)


def compute_closeness(
    graph: nx.Graph | nx.DiGraph,
    *,
    wf_improved: bool = True,
) -> dict[Any, float]:
    """Compute closeness centrality for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    wf_improved
        Whether to use Wasserman-Faust improvement.

    Returns
    -------
    dict[N, float]
        Node to closeness centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.closeness_centrality(graph, wf_improved=wf_improved)


def compute_degree_centrality(
    graph: nx.Graph | nx.DiGraph,
) -> dict[Any, float]:
    """Compute degree centrality for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[N, float]
        Node to degree centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.degree_centrality(graph)


def compute_in_degree_centrality(graph: nx.DiGraph) -> dict[Any, float]:
    """Compute in-degree centrality for all nodes.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    dict[N, float]
        Node to in-degree centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.in_degree_centrality(graph)


def compute_out_degree_centrality(graph: nx.DiGraph) -> dict[Any, float]:
    """Compute out-degree centrality for all nodes.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    dict[N, float]
        Node to out-degree centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.out_degree_centrality(graph)


def compute_all_centralities(
    graph: nx.DiGraph,
    *,
    alpha: float = 0.85,
    betweenness_k: int | None = None,
) -> dict[Any, CentralityMetrics]:
    """Compute all centrality metrics for all nodes.

    Parameters
    ----------
    graph
        Directed graph.
    alpha
        PageRank damping factor.
    betweenness_k
        Sample size for betweenness (None for exact).

    Returns
    -------
    dict[N, CentralityMetrics]
        Node to centrality metrics mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    pagerank = compute_pagerank(graph, alpha=alpha)
    betweenness = compute_betweenness(graph, k=betweenness_k)
    closeness = compute_closeness(graph)

    result: dict[Any, CentralityMetrics] = {}
    for node in graph.nodes():
        # Cast needed due to imprecise NetworkX stubs
        in_degree = cast("int", graph.in_degree(node))
        out_degree = cast("int", graph.out_degree(node))
        result[node] = CentralityMetrics(
            pagerank=pagerank.get(node, 0.0),
            betweenness=betweenness.get(node, 0.0),
            closeness=closeness.get(node, 0.0),
            in_degree=in_degree,
            out_degree=out_degree,
            degree=in_degree + out_degree,
        )
    return result


def centrality_to_rows(
    metrics: Mapping[int, CentralityMetrics],
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    """Convert centrality metrics to row dictionaries.

    Parameters
    ----------
    metrics
        Node to centrality metrics mapping.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    list[dict[str, object]]
        Row dictionaries for persistence.
    """
    return [
        {
            "goid_h128": node,
            "repo": repo,
            "commit": commit,
            "pagerank": m.pagerank,
            "betweenness": m.betweenness,
            "closeness": m.closeness,
            "in_degree": m.in_degree,
            "out_degree": m.out_degree,
            "degree": m.degree,
        }
        for node, m in metrics.items()
    ]


__all__ = [
    "CentralityMetrics",
    "centrality_to_rows",
    "compute_all_centralities",
    "compute_betweenness",
    "compute_closeness",
    "compute_degree_centrality",
    "compute_in_degree_centrality",
    "compute_out_degree_centrality",
    "compute_pagerank",
]
