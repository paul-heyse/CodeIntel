"""Bipartite graph projection and metric computation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx

from codeintel.build.graphs.compute.metrics.bipartite import (
    compute_bipartite_degrees,
    compute_weighted_projection,
)
from codeintel.build.graphs.compute.metrics.centrality import _betweenness_sample
from codeintel.build.graphs.compute.metrics.community import detect_communities_greedy
from codeintel.build.graphs.compute.metrics.conversions import log_projection_skipped
from codeintel.build.graphs.compute.metrics.structural import compute_clustering_coefficient
from codeintel.build.graphs.compute.metrics.types import BipartiteDegrees, ProjectionMetrics
from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, graph_node_count
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, sorted_mapping
from codeintel.core.compute.centrality import compute_betweenness, compute_closeness

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.graphs.runtime.context import GraphContext

log = logging.getLogger(__name__)


_MAX_EDGES_FOR_FULL_METRICS = 50_000


def build_projection_graph(
    bipartite_graph: GraphInput,
    nodes: Iterable[Any],
    *,
    label: str,
) -> nx.Graph:
    """Build a weighted projection graph from a bipartite partition.

    Returns
    -------
    nx.Graph
        Weighted projection graph for the requested partition.
    """
    nodes_set = set(nodes)
    store = ensure_store(bipartite_graph)
    graph_nodes = store.graph.num_nodes()

    result = compute_weighted_projection(store, nodes_set)
    if result is None:
        if not nodes_set:
            reason = "empty partition"
        elif not nodes_set.issubset(set(store.node_ids())):
            reason = "nodes not in graph"
        elif len(nodes_set) >= graph_nodes:
            reason = "partition too large"
        else:
            reason = "projection failure"
        log_projection_skipped(
            label,
            reason,
            nodes=len(nodes_set),
            graph_nodes=graph_nodes,
        )
        return nx.Graph()
    return result


def community_ids(graph: GraphInput, *, weight: str | None = "weight") -> dict[Any, int]:
    """Compute community ids using greedy modularity.

    Returns
    -------
    dict[Any, int]
        Mapping of node to community id.
    """
    return detect_communities_greedy(graph, weight=weight)


def projection_metrics(
    bipartite_graph: GraphInput,
    nodes: Iterable[Any],
    ctx: GraphContext,
    *,
    projection: GraphInput | None = None,
    label: str = "projection",
) -> ProjectionMetrics:
    """Compute weighted projection metrics for a bipartite partition.

    Returns
    -------
    ProjectionMetrics
        Projection metric bundle.
    """
    weight_attr = ctx.pagerank_weight if ctx.pagerank_weight is not None else "weight"
    proj = projection if projection is not None else build_projection_graph(
        bipartite_graph, nodes, label=label
    )
    if graph_node_count(proj) == 0:
        return ProjectionMetrics(
            degree={},
            weighted_degree={},
            clustering={},
            betweenness={},
            closeness={},
            community_id={},
        )

    proj_store = ensure_store(proj, weight=weight_attr)
    node_count = proj_store.graph.num_nodes()
    edge_count = proj_store.graph.num_edges()
    log.info(
        "projection_metrics.start label=%s nodes=%d edges=%d",
        label or "unnamed",
        node_count,
        edge_count,
    )

    degree: dict[Any, int] = dict.fromkeys(proj_store.node_ids(), 0)
    weighted_degree: dict[Any, float] = dict.fromkeys(proj_store.node_ids(), 0.0)
    for src_idx, dst_idx in proj_store.graph.edge_list():
        src_id = proj_store.index_to_id[src_idx]
        dst_id = proj_store.index_to_id[dst_idx]
        payload = proj_store.graph.get_edge_data(src_idx, dst_idx)
        weight_val = edge_weight_from_payload(payload)
        if src_idx == dst_idx:
            degree[src_id] += 2
            weighted_degree[src_id] += weight_val * 2.0
            continue
        degree[src_id] += 1
        degree[dst_id] += 1
        weighted_degree[src_id] += weight_val
        weighted_degree[dst_id] += weight_val

    if edge_count > _MAX_EDGES_FOR_FULL_METRICS:
        log.warning(
            "projection_metrics.skip_expensive label=%s edges=%d threshold=%d - "
            "Projection too large for full metrics. Skipping clustering, betweenness, "
            "closeness, and community detection. These columns will be 0.0 in output.",
            label or "unnamed",
            edge_count,
            _MAX_EDGES_FOR_FULL_METRICS,
        )
        return ProjectionMetrics(
            degree=sorted_mapping(degree),
            weighted_degree=sorted_mapping(weighted_degree),
            clustering={},
            betweenness={},
            closeness={},
            community_id={},
        )

    log.debug("projection_metrics.clustering label=%s", label or "unnamed")
    clustering = compute_clustering_coefficient(proj_store, weight=weight_attr)

    log.debug("projection_metrics.betweenness label=%s", label or "unnamed")
    betweenness = compute_betweenness(
        proj_store,
        k=_betweenness_sample(proj, ctx),
        weight=weight_attr,
        seed=ctx.seed,
    )

    log.debug("projection_metrics.closeness label=%s", label or "unnamed")
    closeness = compute_closeness(proj_store)

    log.debug("projection_metrics.community label=%s", label or "unnamed")
    communities = community_ids(proj_store, weight=weight_attr)

    log.info("projection_metrics.complete label=%s", label or "unnamed")
    return ProjectionMetrics(
        degree=sorted_mapping(degree),
        weighted_degree=sorted_mapping(weighted_degree),
        clustering=clustering,
        betweenness=betweenness,
        closeness=closeness,
        community_id=communities,
    )


def bipartite_degrees(
    graph: GraphInput,
    primary: set[Any],
    secondary: set[Any],
    *,
    weight: str | None = "weight",
) -> BipartiteDegrees:
    """Compute degree metrics for bipartite graphs and their projection.

    Returns
    -------
    BipartiteDegrees
        Degree and centrality metrics for both partitions.
    """
    result = compute_bipartite_degrees(graph, primary, secondary, weight=weight)
    return BipartiteDegrees(
        degree=result.degree,
        weighted_degree=result.weighted_degree,
        primary_degree_centrality=result.primary_degree_centrality,
        secondary_degree_centrality=result.secondary_degree_centrality,
    )


__all__ = [
    "bipartite_degrees",
    "build_projection_graph",
    "community_ids",
    "projection_metrics",
]
