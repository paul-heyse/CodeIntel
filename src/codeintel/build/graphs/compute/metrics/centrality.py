"""Centrality computation functions for directed and undirected graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.build.graphs.compute.metrics.types import CentralityBundle, NeighborStats
from codeintel.build.graphs.rx.algos import (
    BetweennessOptions,
    GraphAlgoConfig,
    GraphInput,
    PagerankOptions,
    constraint_by_id,
    ensure_store,
    graph_node_count,
)
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.core.compute.centrality import (
    compute_betweenness,
    compute_closeness,
    compute_eigenvector_centrality,
    compute_harmonic_centrality,
    compute_pagerank,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.graphs.runtime.context import GraphContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentralityComputations:
    """Optional computation overrides for centrality helpers."""

    eigen_fn: Callable[..., dict[Any, float]] | None = None
    constraint_fn: Callable[..., Any] | None = None


def _algo_config(ctx: GraphContext) -> GraphAlgoConfig:
    return GraphAlgoConfig(
        parallel_threshold=ctx.parallel_threshold,
        rayon_threads=ctx.rayon_threads,
        weight_semantics=ctx.weight_semantics,
    )


def _betweenness_sample(graph: GraphInput, ctx: GraphContext) -> int | None:
    """Determine betweenness sampling parameter.

    Returns
    -------
    int | None
        Sample size for betweenness, or None for full computation.
    """
    node_count = graph_node_count(graph)
    if node_count == 0:
        return None
    if node_count <= ctx.betweenness_sample:
        return None
    return ctx.betweenness_sample


def _coerce_edge_weight(value: object) -> int:
    return int(edge_weight_from_payload(value))


def neighbor_stats(graph: GraphInput, *, weight: str | None = None) -> NeighborStats:
    """Accumulate neighbor sets and weighted edge counts.

    Returns
    -------
    NeighborStats
        Neighbor sets and weighted counts for each node.
    """
    store = ensure_store(graph, weight=weight)
    in_neighbors: dict[Any, set[Any]] = {}
    out_neighbors: dict[Any, set[Any]] = {}
    in_counts: dict[Any, int] = {}
    out_counts: dict[Any, int] = {}

    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id[src_idx]
        dst_id = store.index_to_id[dst_idx]
        payload = store.graph.get_edge_data(src_idx, dst_idx)
        weight_val = _coerce_edge_weight(payload)
        out_neighbors.setdefault(src_id, set()).add(dst_id)
        in_neighbors.setdefault(dst_id, set()).add(src_id)
        out_counts[src_id] = out_counts.get(src_id, 0) + weight_val
        in_counts[dst_id] = in_counts.get(dst_id, 0) + weight_val
    return NeighborStats(
        in_neighbors=in_neighbors,
        out_neighbors=out_neighbors,
        in_counts=in_counts,
        out_counts=out_counts,
    )


def centrality_directed(
    graph: GraphInput,
    ctx: GraphContext,
    *,
    weight: str | None = None,
    include_eigen: bool = False,
    compute_overrides: CentralityComputations | None = None,
) -> CentralityBundle:
    """Compute centrality metrics on a directed graph with shared defaults.

    Returns
    -------
    CentralityBundle
        Centrality metrics for the directed graph.
    """
    betweenness_weight = ctx.betweenness_weight if weight is None else weight
    pagerank_weight = ctx.pagerank_weight if weight is None else weight
    algo_config = _algo_config(ctx)

    betweenness = compute_betweenness(
        graph,
        options=BetweennessOptions(
            k=_betweenness_sample(graph, ctx),
            weight=betweenness_weight,
            seed=ctx.seed,
        ),
        algo_config=algo_config,
    )
    closeness = compute_closeness(graph, algo_config=algo_config)
    harmonic = compute_harmonic_centrality(graph, algo_config=algo_config)
    pagerank = compute_pagerank(
        graph,
        options=PagerankOptions(weight=pagerank_weight),
        algo_config=algo_config,
    )

    eigenvector: dict[Any, float] = {}
    if include_eigen and graph_node_count(graph) > 0:
        overrides = compute_overrides or CentralityComputations()
        eigen_fn = overrides.eigen_fn or compute_eigenvector_centrality
        eigenvector = eigen_fn(
            graph,
            max_iter=ctx.eigen_max_iter,
            weight=weight,
            algo_config=algo_config,
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
    graph: GraphInput,
    ctx: GraphContext,
    *,
    weight: str | None = None,
    include_structural: bool = False,
    compute_overrides: CentralityComputations | None = None,
) -> CentralityBundle:
    """Compute centrality metrics on an undirected graph.

    Returns
    -------
    CentralityBundle
        Centrality metrics for the undirected graph.
    """
    algo_config = _algo_config(ctx)
    betweenness = compute_betweenness(
        graph,
        options=BetweennessOptions(
            k=_betweenness_sample(graph, ctx),
            weight=weight,
            seed=ctx.seed,
        ),
        algo_config=algo_config,
    )
    closeness = compute_closeness(graph, algo_config=algo_config)
    harmonic = compute_harmonic_centrality(graph, algo_config=algo_config)
    pagerank = compute_pagerank(
        graph,
        options=PagerankOptions(weight=weight),
        algo_config=algo_config,
    )

    eigenvector: dict[Any, float] = {}
    if graph_node_count(graph) > 0:
        overrides = compute_overrides or CentralityComputations()
        eigen_fn = overrides.eigen_fn or compute_eigenvector_centrality
        eigenvector = eigen_fn(
            graph,
            max_iter=ctx.eigen_max_iter,
            weight=weight,
            algo_config=algo_config,
        )
        if not eigenvector:
            log.warning("Eigenvector centrality did not converge for graph=%s", graph)

    if include_structural and graph_node_count(graph) > 0:
        overrides = compute_overrides or CentralityComputations()
        try:
            constraint_fn = overrides.constraint_fn or constraint_by_id
            constraint_fn(graph)
        except (TypeError, ValueError):
            log.warning("Structural holes calculation failed for graph=%s", graph)

    return CentralityBundle(
        pagerank=pagerank,
        betweenness=betweenness,
        closeness=closeness,
        harmonic=harmonic,
        eigenvector=eigenvector,
    )


__all__ = [
    "CentralityComputations",
    "centrality_directed",
    "centrality_undirected",
    "neighbor_stats",
]
