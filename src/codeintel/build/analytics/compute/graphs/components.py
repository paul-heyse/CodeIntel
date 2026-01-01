"""Component analysis functions for graphs.

This module provides functions for computing component metadata,
global graph statistics, and layer analysis.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from codeintel.build.analytics.compute.graphs.types import ComponentBundle, GlobalGraphStats
from codeintel.build.graphs.compute.metrics.components import (
    find_connected,
    find_strongly_connected,
    find_weakly_connected,
    topological_layers,
)
from codeintel.build.graphs.compute.metrics.statistics import (
    compute_avg_shortest_path_length,
    compute_condensation_layer_count,
    compute_diameter_estimate,
)
from codeintel.build.graphs.compute.metrics.structural import compute_clustering_coefficient


def component_metadata(graph: nx.DiGraph) -> ComponentBundle:
    """Return weak component, SCC, cycle, and layer metadata.

    Parameters
    ----------
    graph
        Directed graph from which to derive connectivity metadata.

    Returns
    -------
    ComponentBundle
        Component identifiers, sizes, cycle membership, and condensation layers.
    """
    if graph.number_of_nodes() == 0:
        return ComponentBundle(
            component_id={},
            component_size={},
            scc_id={},
            scc_size={},
            in_cycle={},
            layer={},
        )

    weak_infos = find_weakly_connected(graph)
    component_id: dict[Any, int] = {}
    component_size: dict[Any, int] = {}
    for info in weak_infos:
        for node in info.nodes:
            component_id[node] = info.component_id
            component_size[node] = info.size

    scc_result = find_strongly_connected(graph, compute_condensation=True)
    scc_id: dict[Any, int] = scc_result.node_to_component
    scc_size: dict[Any, int] = {}
    for comp in scc_result.components:
        for node in comp.nodes:
            scc_size[node] = comp.size
    in_cycle = {node: scc_size.get(node, 1) > 1 for node in graph.nodes}

    layer_map: dict[Any, int] = {}
    if scc_result.condensation is not None:
        condensation_layer = topological_layers(scc_result.condensation)
        layer_map = {node: condensation_layer.get(scc_id.get(node, 0), 0) for node in graph.nodes}

    return ComponentBundle(
        component_id=component_id,
        component_size=component_size,
        scc_id=scc_id,
        scc_size=scc_size,
        in_cycle=in_cycle,
        layer=layer_map,
    )


def component_ids_undirected(graph: nx.Graph) -> tuple[dict[Any, int], dict[Any, int]]:
    """Return component ids and sizes for undirected graphs.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    tuple[dict[Any, int], dict[Any, int]]
        Component id and size mappings.
    """
    if graph.number_of_nodes() == 0:
        return {}, {}

    comp_infos = find_connected(graph)
    component_id: dict[Any, int] = {}
    component_size: dict[Any, int] = {}
    for info in comp_infos:
        for node in info.nodes:
            component_id[node] = info.component_id
            component_size[node] = info.size

    return component_id, component_size


def _component_layers(graph: nx.Graph | nx.DiGraph) -> int | None:
    """Return the number of condensation layers for directed graphs.

    Parameters
    ----------
    graph
        Graph to analyze.

    Returns
    -------
    int | None
        Layer count for directed graphs; otherwise ``None``.
    """
    if not isinstance(graph, nx.DiGraph):
        return None
    return compute_condensation_layer_count(graph)


def _diameter_and_spl(graph: nx.Graph | nx.DiGraph) -> tuple[float | None, float | None]:
    """Compute diameter and average shortest path length for a graph.

    Parameters
    ----------
    graph
        Graph to analyze.

    Returns
    -------
    tuple[float | None, float | None]
        Diameter and average shortest path length of largest connected component.
    """
    diameter = compute_diameter_estimate(graph)
    avg_spl = compute_avg_shortest_path_length(graph)
    return diameter, avg_spl


def global_graph_stats(graph: nx.Graph | nx.DiGraph) -> GlobalGraphStats:
    """Return global statistics for the provided graph.

    Parameters
    ----------
    graph
        Graph to evaluate.

    Returns
    -------
    GlobalGraphStats
        Counts and structural aggregates.
    """
    diameter_estimate, avg_spl_estimate = _diameter_and_spl(graph)
    component_layers = _component_layers(graph)

    clustering_map = compute_clustering_coefficient(graph)
    avg_clustering = sum(clustering_map.values()) / len(clustering_map) if clustering_map else 0.0

    if isinstance(graph, nx.DiGraph):
        weak_infos = find_weakly_connected(graph)
        weak_component_count = len(weak_infos)
        scc_result = find_strongly_connected(graph)
        scc_count = len(scc_result.components)
    else:
        conn_infos = find_connected(graph)
        weak_component_count = len(conn_infos)
        scc_count = weak_component_count

    return GlobalGraphStats(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        weak_component_count=weak_component_count,
        scc_count=scc_count,
        component_layers=component_layers,
        avg_clustering=avg_clustering,
        diameter_estimate=diameter_estimate,
        avg_shortest_path_estimate=avg_spl_estimate,
    )


__all__ = [
    "component_ids_undirected",
    "component_metadata",
    "global_graph_stats",
]
