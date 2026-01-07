"""Pure centrality metric computation functions.

This module provides stateless functions for computing centrality metrics
on NetworkX graphs without any database or file I/O.

These functions are the canonical implementations used by both the graphs
and analytics packages.

Functions
---------
compute_pagerank
    Compute PageRank scores for all nodes.
compute_betweenness
    Compute betweenness centrality.
compute_closeness
    Compute closeness centrality.
compute_harmonic_centrality
    Compute harmonic centrality.
compute_eigenvector_centrality
    Compute eigenvector centrality.
compute_degree_centrality
    Compute degree centrality.
compute_in_degree_centrality
    Compute in-degree centrality (directed graphs).
compute_out_degree_centrality
    Compute out-degree centrality (directed graphs).
compute_all_centralities
    Compute all centrality metrics in one pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import rustworkx as rx

from codeintel.build.graphs.rx.algos import (
    BetweennessOptions,
    GraphInput,
    PagerankOptions,
    betweenness_by_id,
    closeness_by_id,
    degree_centrality_by_id,
    eigenvector_centrality_by_id,
    ensure_store,
    harmonic_centrality_by_id,
    in_degree_centrality_by_id,
    out_degree_centrality_by_id,
    pagerank_by_id,
)

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
    graph: GraphInput,
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
    store = ensure_store(graph, weight=weight)
    if store.graph.num_nodes() == 0:
        return {}
    try:
        return pagerank_by_id(
            store,
            options=PagerankOptions(
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                weight=weight,
            ),
        )
    except rx.FailedToConverge:
        node_count = store.graph.num_nodes()
        if node_count == 0:
            return {}
        return dict.fromkeys(store.node_ids(), 1.0 / node_count)


def compute_betweenness(
    graph: GraphInput,
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
    store = ensure_store(graph, weight=weight)
    if store.graph.num_nodes() == 0:
        return {}
    return betweenness_by_id(
        store,
        options=BetweennessOptions(
            normalized=normalized,
            k=k,
            weight=weight,
            seed=seed,
        ),
    )


def compute_closeness(
    graph: GraphInput,
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
    dict[Any, float]
        Node to closeness centrality mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return closeness_by_id(store, wf_improved=wf_improved)


def compute_degree_centrality(
    graph: GraphInput,
) -> dict[Any, float]:
    """Compute degree centrality for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).

    Returns
    -------
    dict[Any, float]
        Node to degree centrality mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return degree_centrality_by_id(store)


def compute_in_degree_centrality(graph: GraphInput) -> dict[Any, float]:
    """Compute in-degree centrality for all nodes.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    dict[Any, float]
        Node to in-degree centrality mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return in_degree_centrality_by_id(store)


def compute_out_degree_centrality(graph: GraphInput) -> dict[Any, float]:
    """Compute out-degree centrality for all nodes.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    dict[Any, float]
        Node to out-degree centrality mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return out_degree_centrality_by_id(store)


def compute_harmonic_centrality(
    graph: GraphInput,
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
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return harmonic_centrality_by_id(store)


def compute_eigenvector_centrality(
    graph: GraphInput,
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
    store = ensure_store(graph, weight=weight)
    if store.graph.num_nodes() == 0:
        return {}
    try:
        return eigenvector_centrality_by_id(
            store,
            max_iter=max_iter,
            tol=tol,
            weight=weight,
        )
    except rx.FailedToConverge:
        log.warning("Eigenvector centrality did not converge; returning zeros")
        return dict.fromkeys(store.node_ids(), 0.0)


def compute_all_centralities(
    graph: GraphInput,
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
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}

    pagerank = compute_pagerank(store, alpha=alpha)
    betweenness = compute_betweenness(store, k=betweenness_k)
    closeness = compute_closeness(store)
    harmonic = compute_harmonic_centrality(store)
    eigenvector: dict[Any, float] = {}
    if include_eigenvector:
        eigenvector = compute_eigenvector_centrality(store, max_iter=eigenvector_max_iter)

    result: dict[Any, CentralityMetrics] = {}
    for node_id in store.node_ids():
        node_idx = store.id_to_index[node_id]
        if store.is_directed:
            in_degree = store.graph.in_degree(node_idx)
            out_degree = store.graph.out_degree(node_idx)
            degree = in_degree + out_degree
        else:
            degree = store.graph.degree(node_idx)
            in_degree = degree
            out_degree = 0
        result[node_id] = CentralityMetrics(
            pagerank=pagerank.get(node_id, 0.0),
            betweenness=betweenness.get(node_id, 0.0),
            closeness=closeness.get(node_id, 0.0),
            harmonic=harmonic.get(node_id, 0.0),
            eigenvector=eigenvector.get(node_id, 0.0),
            in_degree=in_degree,
            out_degree=out_degree,
            degree=degree,
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
