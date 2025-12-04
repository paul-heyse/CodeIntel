"""Centrality computation functions for directed and undirected graphs.

This module provides high-level functions for computing bundles of
centrality metrics on graphs, delegating to pure computation functions
in codeintel.graphs.compute.metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.algorithms import structuralholes
from networkx.exception import NetworkXAlgorithmError

from codeintel.analytics.compute.graphs.types import CentralityBundle, NeighborStats
from codeintel.graphs.compute.metrics.centrality import (
    compute_betweenness,
    compute_closeness,
    compute_eigenvector_centrality,
    compute_harmonic_centrality,
    compute_pagerank,
)

if TYPE_CHECKING:
    from codeintel.analytics.graphs.runtime import GraphContext

log = logging.getLogger(__name__)


def _betweenness_sample(graph: nx.Graph, ctx: GraphContext) -> int | None:
    """Determine betweenness sampling parameter.

    Parameters
    ----------
    graph
        Graph to analyze.
    ctx
        Graph context with sampling configuration.

    Returns
    -------
    int | None
        Sample size for betweenness, or None for exact computation.
    """
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return None
    if node_count <= ctx.betweenness_sample:
        return None
    return ctx.betweenness_sample


def neighbor_stats(graph: nx.DiGraph, *, weight: str | None = None) -> NeighborStats:
    """Accumulate neighbor sets and weighted edge counts.

    Parameters
    ----------
    graph
        Directed graph containing edges to evaluate.
    weight
        Edge attribute storing the weight. When None, uses "weight" if present else 1.

    Returns
    -------
    NeighborStats
        Aggregated inbound/outbound neighbor sets and weighted degree counts.
    """
    in_neighbors: dict[Any, set[Any]] = {}
    out_neighbors: dict[Any, set[Any]] = {}
    in_counts: dict[Any, int] = {}
    out_counts: dict[Any, int] = {}

    for src, dst, data in graph.edges(data=True):
        key = "weight" if weight is None else weight
        weight_val = int(data.get(key, 1)) if key is not None else 1
        out_neighbors.setdefault(src, set()).add(dst)
        in_neighbors.setdefault(dst, set()).add(src)
        out_counts[src] = out_counts.get(src, 0) + weight_val
        in_counts[dst] = in_counts.get(dst, 0) + weight_val
    return NeighborStats(
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        in_counts=in_counts,
        out_counts=out_counts,
    )


def centrality_directed(
    graph: nx.DiGraph,
    ctx: GraphContext,
    *,
    weight: str | None = None,
    include_eigen: bool = False,
) -> CentralityBundle:
    """Compute centrality metrics on a directed graph with shared defaults.

    Parameters
    ----------
    graph
        Directed graph to evaluate.
    ctx
        Execution context controlling sampling, iteration limits, and seeds.
    weight
        Edge attribute storing the weight. Defaults to context betweenness/pagerank weight.
    include_eigen
        Whether to compute eigenvector centrality on an undirected view.

    Returns
    -------
    CentralityBundle
        PageRank, betweenness, closeness, harmonic, and optional eigenvector scores.
    """
    betweenness_weight = ctx.betweenness_weight if weight is None else weight
    pagerank_weight = ctx.pagerank_weight if weight is None else weight

    betweenness = compute_betweenness(
        graph,
        k=_betweenness_sample(graph, ctx),
        weight=betweenness_weight,
        seed=ctx.seed,
    )
    closeness = compute_closeness(graph)
    harmonic = compute_harmonic_centrality(graph)
    pagerank = compute_pagerank(graph, weight=pagerank_weight)

    eigenvector: dict[Any, float] = {}
    if include_eigen and graph.number_of_nodes() > 0:
        eigenvector = compute_eigenvector_centrality(
            graph, max_iter=ctx.eigen_max_iter, weight=weight
        )
        if not eigenvector:
            log.warning("Eigenvector centrality did not converge for graph=%s", graph)

    return CentralityBundle(
        pagerank=pagerank,
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigenvector=eigenvector,
    )


def centrality_undirected(
    graph: nx.Graph,
    ctx: GraphContext,
    *,
    weight: str | None = None,
    include_structural: bool = False,
) -> CentralityBundle:
    """Compute centrality metrics on an undirected graph.

    Parameters
    ----------
    graph
        Undirected graph to evaluate.
    ctx
        Execution context controlling sampling, iteration limits, and seeds.
    weight
        Edge attribute storing the weight. Defaults to "weight".
    include_structural
        Whether to compute additional structural hole metrics.

    Returns
    -------
    CentralityBundle
        PageRank, betweenness, closeness, harmonic, and eigenvector scores.
    """
    betweenness_weight = ctx.betweenness_weight if weight is None else weight
    pagerank_weight = ctx.pagerank_weight if weight is None else weight

    betweenness = compute_betweenness(
        graph,
        k=_betweenness_sample(graph, ctx),
        weight=betweenness_weight,
        seed=ctx.seed,
    )
    closeness = compute_closeness(graph)
    harmonic = compute_harmonic_centrality(graph)
    pagerank = compute_pagerank(graph, weight=pagerank_weight)

    eigenvector: dict[Any, float] = {}
    if graph.number_of_nodes() > 0:
        eigenvector = compute_eigenvector_centrality(
            graph, max_iter=ctx.eigen_max_iter, weight=pagerank_weight
        )
        if not eigenvector:
            log.warning("Eigenvector centrality did not converge for undirected graph=%s", graph)

    if include_structural and graph.number_of_nodes() > 0:
        try:
            _ = structuralholes.constraint(graph, weight=weight)
        except NetworkXAlgorithmError:
            log.warning("Structural holes calculation failed for graph=%s", graph)

    return CentralityBundle(
        pagerank=pagerank,
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigenvector=eigenvector,
    )


__all__ = [
    "centrality_directed",
    "centrality_undirected",
    "neighbor_stats",
]
