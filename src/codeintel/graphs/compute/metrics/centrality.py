"""Pure centrality metric computation functions.

This module provides stateless functions for computing centrality metrics
on NetworkX graphs without any database or file I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


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
    harmonic
        Harmonic centrality.
    eigenvector
        Eigenvector centrality.
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
    harmonic: float
    eigenvector: float
    in_degree: int
    out_degree: int
    degree: int


def compute_pagerank(
    graph: nx.DiGraph | nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight: str | None = None,
) -> dict[Any, float]:
    """Compute PageRank for all nodes in a graph.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    alpha
        Damping factor.
    max_iter
        Maximum iterations.
    tol
        Convergence tolerance.
    weight
        Edge attribute to use as weight (None for unweighted).

    Returns
    -------
    dict[Any, float]
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
        return {
            node: float(val)
            for node, val in nx.pagerank(
                graph, alpha=alpha, max_iter=max_iter, tol=tol, weight=weight
            ).items()
        }
    except nx.PowerIterationFailedConvergence:
        n = graph.number_of_nodes()
        return dict.fromkeys(graph.nodes(), 1.0 / n)


def compute_betweenness(
    graph: nx.Graph | nx.DiGraph,
    *,
    normalized: bool = True,
    k: int | None = None,
    weight: str | None = None,
    seed: int | None = None,
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
    weight
        Edge attribute to use as weight (None for unweighted).
    seed
        Random seed for sampling (used when k is specified).

    Returns
    -------
    dict[Any, float]
        Node to betweenness centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return {
        node: float(val)
        for node, val in nx.betweenness_centrality(
            graph, normalized=normalized, k=k, weight=weight, seed=seed
        ).items()
    }


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


def compute_harmonic_centrality(
    graph: nx.Graph | nx.DiGraph,
) -> dict[Any, float]:
    """Compute harmonic centrality for all nodes.

    Harmonic centrality is more robust to disconnected graphs than
    closeness centrality as it uses the sum of reciprocal distances.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, float]
        Node to harmonic centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return {node: float(val) for node, val in nx.harmonic_centrality(graph).items()}


def compute_eigenvector_centrality(
    graph: nx.Graph | nx.DiGraph,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight: str | None = None,
) -> dict[Any, float]:
    """Compute eigenvector centrality for all nodes.

    For directed graphs, computes on the undirected view to ensure
    convergence.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    max_iter
        Maximum iterations for power iteration.
    tol
        Convergence tolerance.
    weight
        Edge attribute to use as weight (None for unweighted).

    Returns
    -------
    dict[Any, float]
        Node to eigenvector centrality mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    work_graph = graph.to_undirected() if isinstance(graph, nx.DiGraph) else graph

    try:
        return {
            node: float(val)
            for node, val in nx.eigenvector_centrality(
                work_graph,
                max_iter=max_iter,
                tol=tol,
                weight=weight,
            ).items()
        }
    except nx.PowerIterationFailedConvergence:
        log.warning("Eigenvector centrality did not converge; returning zeros")
        return dict.fromkeys(graph.nodes(), 0.0)


def compute_all_centralities(
    graph: nx.DiGraph,
    *,
    alpha: float = 0.85,
    betweenness_k: int | None = None,
    include_eigenvector: bool = True,
    eigenvector_max_iter: int = 100,
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
    include_eigenvector
        Whether to compute eigenvector centrality (can be slow/fail to converge).
    eigenvector_max_iter
        Maximum iterations for eigenvector computation.

    Returns
    -------
    dict[Any, CentralityMetrics]
        Node to centrality metrics mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    pagerank = compute_pagerank(graph, alpha=alpha)
    betweenness = compute_betweenness(graph, k=betweenness_k)
    closeness = compute_closeness(graph)
    harmonic = compute_harmonic_centrality(graph)
    eigenvector: dict[Any, float] = {}
    if include_eigenvector:
        eigenvector = compute_eigenvector_centrality(graph, max_iter=eigenvector_max_iter)

    result: dict[Any, CentralityMetrics] = {}
    for node in graph.nodes():
        in_degree = cast("int", graph.in_degree(node))
        out_degree = cast("int", graph.out_degree(node))
        result[node] = CentralityMetrics(
            pagerank=pagerank.get(node, 0.0),
            betweenness=betweenness.get(node, 0.0),
            closeness=closeness.get(node, 0.0),
            harmonic=harmonic.get(node, 0.0),
            eigenvector=eigenvector.get(node, 0.0),
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
            "harmonic": m.harmonic,
            "eigenvector": m.eigenvector,
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
    "compute_eigenvector_centrality",
    "compute_harmonic_centrality",
    "compute_in_degree_centrality",
    "compute_out_degree_centrality",
    "compute_pagerank",
]
