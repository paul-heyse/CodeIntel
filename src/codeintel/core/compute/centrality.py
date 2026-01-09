"""Pure centrality metric computation functions.

This module provides stateless functions for computing centrality metrics
on rustworkx graph stores without any database or file I/O.

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
from typing import TYPE_CHECKING, Any, cast

import rustworkx as rx

from codeintel.build.graphs.rx.algos import (
    BetweennessOptions,
    EigenvectorOptions,
    GraphAlgoConfig,
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


@dataclass(frozen=True, slots=True)
class AllCentralitiesOptions:
    """Configuration for computing the full centrality suite."""

    alpha: float = 0.85
    betweenness_k: int | None = None
    include_eigenvector: bool = True
    eigenvector_max_iter: int = 100


def compute_pagerank(
    graph: GraphInput,
    *,
    options: PagerankOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Any, float]:
    """Compute PageRank for all nodes in a graph.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    options
        PageRank options (alpha, weight, convergence settings).
    algo_config
        Optional algorithm runtime configuration (parallelism, weight semantics).

    Returns
    -------
    dict[Any, float]
        Node to PageRank score mapping.

    Examples
    --------
    >>> from codeintel.build.graphs.rx.store import RxGraphStore
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> g.add_weighted_edge(2, 3, weight=1.0)
    >>> g.add_weighted_edge(3, 1, weight=1.0)
    >>> pr = compute_pagerank(g)
    >>> len(pr)
    3
    """
    resolved = options or PagerankOptions()
    store = ensure_store(graph, weight=resolved.weight)
    if store.graph.num_nodes() == 0:
        return {}
    try:
        return pagerank_by_id(
            store,
            options=resolved,
            algo_config=algo_config,
        )
    except rx.FailedToConverge:
        node_count = store.graph.num_nodes()
        if node_count == 0:
            return {}
        return dict.fromkeys(store.node_ids(), 1.0 / node_count)


def compute_betweenness(
    graph: GraphInput,
    *,
    options: BetweennessOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Any, float]:
    """Compute betweenness centrality for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    options
        Betweenness computation options (normalization, sampling, weighting).
    algo_config
        Optional algorithm runtime configuration (parallelism, weight semantics).

    Returns
    -------
    dict[Any, float]
        Node to betweenness centrality mapping.
    """
    resolved = options or BetweennessOptions()
    store = ensure_store(graph, weight=resolved.weight)
    if store.graph.num_nodes() == 0:
        return {}
    return betweenness_by_id(
        store,
        options=resolved,
        algo_config=algo_config,
    )


def compute_closeness(
    graph: GraphInput,
    *,
    wf_improved: bool = True,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Any, float]:
    """Compute closeness centrality for all nodes.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    wf_improved
        Whether to use Wasserman-Faust improvement.
    algo_config
        Optional algorithm runtime configuration (parallelism, weight semantics).

    Returns
    -------
    dict[Any, float]
        Node to closeness centrality mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return closeness_by_id(store, wf_improved=wf_improved, algo_config=algo_config)


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
    *,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Any, float]:
    """Compute harmonic centrality for all nodes.

    Harmonic centrality is more robust to disconnected graphs than
    closeness centrality as it uses the sum of reciprocal distances.

    Parameters
    ----------
    graph
        Graph (directed or undirected).
    algo_config
        Optional algorithm runtime configuration (parallelism, weight semantics).

    Returns
    -------
    dict[Any, float]
        Node to harmonic centrality mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    return harmonic_centrality_by_id(store, algo_config=algo_config)


def compute_eigenvector_centrality(
    graph: GraphInput,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight: str | None = None,
    algo_config: GraphAlgoConfig | None = None,
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
    algo_config
        Optional algorithm runtime configuration (parallelism, weight semantics).

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
            options=EigenvectorOptions(
                max_iter=max_iter,
                tol=tol,
                weight=weight,
            ),
            algo_config=algo_config,
        )
    except rx.FailedToConverge:
        log.warning("Eigenvector centrality did not converge; returning zeros")
        return dict.fromkeys(store.node_ids(), 0.0)


def compute_all_centralities(
    graph: GraphInput,
    *,
    options: AllCentralitiesOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Any, CentralityMetrics]:
    """Compute all centrality metrics for all nodes.

    Parameters
    ----------
    graph
        Directed graph.
    options
        Configuration for which metrics to compute and their defaults.
    algo_config
        Optional algorithm runtime configuration (parallelism, weight semantics).

    Returns
    -------
    dict[Any, CentralityMetrics]
        Node to centrality metrics mapping.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}

    resolved = options or AllCentralitiesOptions()
    pagerank = compute_pagerank(
        store,
        options=PagerankOptions(alpha=resolved.alpha),
        algo_config=algo_config,
    )
    betweenness = compute_betweenness(
        store,
        options=BetweennessOptions(k=resolved.betweenness_k),
        algo_config=algo_config,
    )
    closeness = compute_closeness(store, algo_config=algo_config)
    harmonic = compute_harmonic_centrality(store, algo_config=algo_config)
    eigenvector: dict[Any, float] = {}
    if resolved.include_eigenvector:
        eigenvector = compute_eigenvector_centrality(
            store,
            max_iter=resolved.eigenvector_max_iter,
            algo_config=algo_config,
        )

    result: dict[Any, CentralityMetrics] = {}
    directed_graph: rx.PyDiGraph | None = None
    undirected_graph: rx.PyGraph | None = None
    if store.is_directed:
        directed_graph = cast("rx.PyDiGraph", store.graph)
    else:
        undirected_graph = cast("rx.PyGraph", store.graph)
    for node_id in store.node_ids():
        node_idx = store.id_to_index[node_id]
        if store.is_directed:
            if directed_graph is None:
                directed_graph = cast("rx.PyDiGraph", store.graph)
            in_degree = directed_graph.in_degree(node_idx)
            out_degree = directed_graph.out_degree(node_idx)
            degree = in_degree + out_degree
        else:
            if undirected_graph is None:
                undirected_graph = cast("rx.PyGraph", store.graph)
            degree = undirected_graph.degree(node_idx)
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
