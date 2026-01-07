"""Pure computation for bipartite graph metrics.

This module provides functions to compute metrics for bipartite graphs,
including degree centrality and weighted projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeintel.build.graphs.rx.algos import (
    GraphInput,
    bipartite_degree_centrality_by_id,
    ensure_store,
    to_undirected_store,
    weighted_projection_store,
)
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, sorted_mapping
from codeintel.build.graphs.rx.store import RxGraphStore


@dataclass(frozen=True)
class BipartiteDegreeMetrics:
    """Degree metrics for bipartite graph partitions.

    Attributes
    ----------
    degree
        Mapping of node to unweighted degree.
    weighted_degree
        Mapping of node to weighted degree.
    primary_degree_centrality
        Degree centrality for primary partition nodes.
    secondary_degree_centrality
        Degree centrality for secondary partition nodes.
    """

    degree: dict[Any, int]
    weighted_degree: dict[Any, float]
    primary_degree_centrality: dict[Any, float]
    secondary_degree_centrality: dict[Any, float]


def compute_bipartite_degrees(
    graph: GraphInput,
    primary: set[Any],
    secondary: set[Any],
    *,
    weight: str | None = "weight",
) -> BipartiteDegreeMetrics:
    """Compute degree metrics for bipartite graph partitions.

    Parameters
    ----------
    graph
        Bipartite graph to analyze.
    primary
        Set of primary partition nodes.
    secondary
        Set of secondary partition nodes.
    weight
        Edge attribute storing weight; defaults to "weight".

    Returns
    -------
    BipartiteDegreeMetrics
        Degree metrics for both partitions.

    Examples
    --------
    >>> g = RxGraphStore.undirected()
    >>> g.add_weighted_edge(1, "a", weight=1.0)
    >>> g.add_weighted_edge(1, "b", weight=1.0)
    >>> g.add_weighted_edge(2, "b", weight=1.0)
    >>> result = compute_bipartite_degrees(g, {1, 2}, {"a", "b"})
    >>> result.degree[1]
    2
    """
    store = ensure_store(graph, weight=weight)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return BipartiteDegreeMetrics(
            degree={},
            weighted_degree={},
            primary_degree_centrality={},
            secondary_degree_centrality={},
        )

    degree: dict[Any, int] = dict.fromkeys(work_store.node_ids(), 0)
    weighted_degree: dict[Any, float] = dict.fromkeys(work_store.node_ids(), 0.0)
    for src_idx, dst_idx in work_store.graph.edge_list():
        src_id = work_store.index_to_id[src_idx]
        dst_id = work_store.index_to_id[dst_idx]
        payload = work_store.graph.get_edge_data(src_idx, dst_idx)
        weight_val = edge_weight_from_payload(payload)
        if src_idx == dst_idx:
            degree[src_id] += 2
            weighted_degree[src_id] += weight_val * 2.0
            continue
        degree[src_id] += 1
        degree[dst_id] += 1
        weighted_degree[src_id] += weight_val
        weighted_degree[dst_id] += weight_val

    if not primary or not secondary:
        return BipartiteDegreeMetrics(
            degree=sorted_mapping(degree),
            weighted_degree=sorted_mapping(weighted_degree),
            primary_degree_centrality={},
            secondary_degree_centrality={},
        )

    centrality = bipartite_degree_centrality_by_id(work_store, primary)
    primary_dc = {node: centrality.get(node, 0.0) for node in primary}
    secondary_dc = {node: centrality.get(node, 0.0) for node in secondary}

    return BipartiteDegreeMetrics(
        degree=sorted_mapping(degree),
        weighted_degree=sorted_mapping(weighted_degree),
        primary_degree_centrality=sorted_mapping(primary_dc),
        secondary_degree_centrality=sorted_mapping(secondary_dc),
    )


def compute_weighted_projection(
    bipartite_graph: GraphInput,
    nodes: set[Any],
) -> RxGraphStore | None:
    """Build a weighted projection graph from a bipartite partition.

    Parameters
    ----------
    bipartite_graph
        Bipartite graph to project.
    nodes
        Set of nodes in the partition to project onto.

    Returns
    -------
    RxGraphStore | None
        Projected graph store, or None if projection cannot be computed.

    Notes
    -----
    The projection fails and returns None if:
    - The nodes set is empty
    - The nodes are not a subset of the graph's nodes
    - The nodes set is equal to or larger than the entire graph

    Examples
    --------
    >>> g = RxGraphStore.undirected()
    >>> g.add_weighted_edge(1, "a", weight=1.0)
    >>> g.add_weighted_edge(1, "b", weight=1.0)
    >>> g.add_weighted_edge(2, "b", weight=1.0)
    >>> proj = compute_weighted_projection(g, {1, 2})
    >>> proj is not None
    True
    >>> proj.graph.num_nodes()
    2
    """
    store = ensure_store(bipartite_graph)
    node_count = store.graph.num_nodes()
    if not nodes:
        return None
    graph_node_set = set(store.node_ids())
    if not nodes.issubset(graph_node_set):
        return None
    if len(nodes) >= node_count:
        return None
    try:
        projection_store = weighted_projection_store(store, nodes, ratio=False)
    except ValueError:
        return None
    return projection_store


__all__ = [
    "BipartiteDegreeMetrics",
    "compute_bipartite_degrees",
    "compute_weighted_projection",
]
