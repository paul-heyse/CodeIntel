"""Bipartite graph projection and metric computation.

This module provides functions for building projections from bipartite
graphs and computing metrics on those projections.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import networkx as nx

from codeintel.analytics.compute.graphs.centrality import _betweenness_sample
from codeintel.analytics.compute.graphs.conversions import log_projection_skipped
from codeintel.analytics.compute.graphs.types import BipartiteDegrees, ProjectionMetrics
from codeintel.graphs.compute.metrics.bipartite import (
    compute_bipartite_degrees,
    compute_weighted_projection,
)
from codeintel.graphs.compute.metrics.community import detect_communities_greedy

if TYPE_CHECKING:
    from codeintel.analytics.runtime.context import GraphContext


def build_projection_graph(
    bipartite_graph: nx.Graph,
    nodes: Iterable[Any],
    *,
    label: str,
) -> nx.Graph:
    """Build a weighted projection graph from a bipartite partition.

    Delegates pure computation to graphs.compute.metrics.bipartite.compute_weighted_projection
    and handles logging/error reporting at the analytics layer.

    Parameters
    ----------
    bipartite_graph
        Bipartite graph to project.
    nodes
        Iterable of nodes in the partition to project onto.
    label
        Label for logging messages.

    Returns
    -------
    nx.Graph
        Projection graph; empty when the projection is skipped.
    """
    nodes_set = set(nodes)
    graph_nodes = bipartite_graph.number_of_nodes()

    result = compute_weighted_projection(bipartite_graph, nodes_set)
    if result is None:
        if not nodes_set:
            reason = "empty partition"
        elif not nodes_set.issubset(set(bipartite_graph)):
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


def community_ids(graph: nx.Graph, *, weight: str | None = "weight") -> dict[Any, int]:
    """Compute community ids using greedy modularity.

    Parameters
    ----------
    graph
        Undirected graph.
    weight
        Edge attribute storing weight. Defaults to "weight".

    Returns
    -------
    dict[Any, int]
        Mapping of node to community id.
    """
    return detect_communities_greedy(graph, weight=weight)


def projection_metrics(
    bipartite_graph: nx.Graph,
    nodes: Iterable[Any],
    ctx: GraphContext,
    *,
    projection: nx.Graph | None = None,
    label: str = "projection",
) -> ProjectionMetrics:
    """Compute weighted projection metrics for a bipartite partition.

    Parameters
    ----------
    bipartite_graph
        Bipartite graph to project.
    nodes
        Nodes in the partition to project onto.
    ctx
        Graph context for sampling parameters.
    projection
        Pre-computed projection graph (optional).
    label
        Label for logging messages.

    Returns
    -------
    ProjectionMetrics
        Degree, weighted degree, clustering, betweenness, closeness, and communities.
    """
    weight_attr = ctx.pagerank_weight if ctx.pagerank_weight is not None else "weight"
    proj = (
        projection
        if projection is not None
        else build_projection_graph(bipartite_graph, nodes, label=label)
    )
    if proj.number_of_nodes() == 0:
        return ProjectionMetrics(
            degree={},
            weighted_degree={},
            clustering={},
            betweenness={},
            closeness={},
            community_id={},
        )
    degree_view = nx.degree(proj, weight=None)
    weighted_view = nx.degree(proj, weight=weight_attr)
    degree = {node: int(deg) for node, deg in degree_view}
    weighted_degree = {node: float(deg) for node, deg in weighted_view}
    clustering_val = nx.clustering(proj, weight=weight_attr) if proj.number_of_nodes() > 0 else {}
    clustering = clustering_val if isinstance(clustering_val, dict) else {}
    betweenness = (
        nx.betweenness_centrality(
            proj,
            weight=weight_attr,
            k=_betweenness_sample(proj, ctx),
            seed=ctx.seed,
        )
        if proj.number_of_nodes() > 0
        else {}
    )
    closeness = {node: float(val) for node, val in nx.closeness_centrality(proj).items()}
    communities = community_ids(proj, weight=weight_attr)
    return ProjectionMetrics(
        degree=degree,
        weighted_degree=weighted_degree,
        clustering=clustering,
        betweenness=betweenness,
        closeness=closeness,
        community_id=communities,
    )


def bipartite_degrees(
    graph: nx.Graph, primary: set[Any], secondary: set[Any], *, weight: str | None = "weight"
) -> BipartiteDegrees:
    """Compute degree metrics for bipartite graphs and their projection.

    Delegates pure computation to graphs.compute.metrics.bipartite.compute_bipartite_degrees.

    Parameters
    ----------
    graph
        Bipartite graph.
    primary
        Set of primary nodes.
    secondary
        Set of secondary nodes.
    weight
        Edge attribute storing weight; defaults to "weight".

    Returns
    -------
    BipartiteDegrees
        Degree metrics for both partitions.
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
