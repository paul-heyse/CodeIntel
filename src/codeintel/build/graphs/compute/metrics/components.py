"""Pure component analysis functions.

This module provides stateless functions for computing graph components
and structural properties without any database or file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from codeintel.build.graphs.compute.metrics.statistics import (
    compute_avg_shortest_path_length,
    compute_condensation_layer_count,
    compute_diameter_estimate,
)
from codeintel.build.graphs.compute.metrics.structural import compute_clustering_coefficient
from codeintel.build.graphs.compute.metrics.types import ComponentBundle, GlobalGraphStats
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    articulation_points_by_id,
    bridges_by_id,
    connected_components_by_id,
    ensure_directed_store,
    ensure_store,
    simple_cycles_by_id,
    strongly_connected_components_by_id,
    topological_layers_by_id,
    weakly_connected_components_by_id,
)
from codeintel.build.graphs.rx.components import invert_membership_map
from codeintel.build.graphs.rx.condensation import condensation_store
from codeintel.build.graphs.rx.store import RxGraphStore


@dataclass(frozen=True)
class ComponentInfo:
    """Information about a connected component.

    Attributes
    ----------
    component_id
        Component identifier.
    size
        Number of nodes in component.
    nodes
        Nodes in the component.
    """

    component_id: int
    size: int
    nodes: frozenset[Any]


@dataclass(frozen=True)
class SCCResult:
    """Result of strongly connected component analysis.

    Attributes
    ----------
    components
        List of SCCs.
    node_to_component
        Node to component ID mapping.
    condensation
        DAG of condensed SCCs (None if not computed).
    """

    components: tuple[ComponentInfo, ...]
    node_to_component: dict[Any, int]
    condensation: RxGraphStore | None = None


class ComponentStats(TypedDict):
    """Summary statistics for a collection of components."""

    count: int
    largest_size: int
    smallest_size: int
    mean_size: float
    singleton_count: int


def _component_infos(components: Sequence[Sequence[Any]]) -> list[ComponentInfo]:
    infos: list[ComponentInfo] = []
    for comp_id, component in enumerate(components):
        nodes = frozenset(component)
        infos.append(ComponentInfo(component_id=comp_id, size=len(nodes), nodes=nodes))
    return infos


def _membership_from_components(components: Sequence[Sequence[Any]]) -> dict[Any, int]:
    membership: dict[Any, int] = {}
    for comp_id, component in enumerate(components):
        for node in component:
            membership[node] = comp_id
    return membership


def find_strongly_connected(
    graph: GraphInput,
    *,
    compute_condensation: bool = False,
) -> SCCResult:
    """Find strongly connected components in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.
    compute_condensation
        Whether to compute the condensation DAG.

    Returns
    -------
    SCCResult
        SCC analysis result.

    Examples
    --------
    >>> g = RxGraphStore.directed()
    >>> g.add_weighted_edge(1, 2, weight=1.0)
    >>> result = find_strongly_connected(g)
    >>> len(result.components) >= 2
    True
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return SCCResult(components=(), node_to_component={})
    if compute_condensation:
        condensed, membership_by_id = condensation_store(store)
        components = invert_membership_map(membership_by_id)
        if not components:
            return SCCResult(components=(), node_to_component={})
        return SCCResult(
            components=tuple(_component_infos(components)),
            node_to_component=membership_by_id,
            condensation=condensed,
        )
    components = strongly_connected_components_by_id(store)
    if not components:
        return SCCResult(components=(), node_to_component={})
    return SCCResult(
        components=tuple(_component_infos(components)),
        node_to_component=_membership_from_components(components),
        condensation=None,
    )


def find_weakly_connected(graph: GraphInput) -> list[ComponentInfo]:
    """Find weakly connected components in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.

    Returns
    -------
    list[ComponentInfo]
        Weakly connected components.
    """
    components = weakly_connected_components_by_id(graph)
    return _component_infos(components)


def find_connected(graph: GraphInput) -> list[ComponentInfo]:
    """Find connected components in an undirected graph.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[ComponentInfo]
        Connected components.
    """
    components = connected_components_by_id(graph)
    return _component_infos(components)


def find_bridges(graph: GraphInput) -> list[tuple[Any, Any]]:
    """Find bridge edges whose removal disconnects the graph.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[tuple[Any, Any]]
        Bridge edges.
    """
    return bridges_by_id(graph)


def find_articulation_points(graph: GraphInput) -> list[Any]:
    """Find articulation points whose removal disconnects the graph.

    Parameters
    ----------
    graph
        Undirected graph.

    Returns
    -------
    list[Any]
        Articulation point nodes.
    """
    return articulation_points_by_id(graph)


def compute_component_stats(
    components: Sequence[ComponentInfo],
) -> ComponentStats:
    """Compute summary statistics for components.

    Parameters
    ----------
    components
        Component information.

    Returns
    -------
    ComponentStats
        Statistics including count, sizes, and largest component.
    """
    if not components:
        return ComponentStats(
            count=0,
            largest_size=0,
            smallest_size=0,
            mean_size=0.0,
            singleton_count=0,
        )

    sizes = [c.size for c in components]
    return ComponentStats(
        count=len(components),
        largest_size=max(sizes),
        smallest_size=min(sizes),
        mean_size=sum(sizes) / len(sizes),
        singleton_count=sum(1 for s in sizes if s == 1),
    )


def find_cycles(graph: GraphInput, limit: int | None = 100) -> list[list[Any]]:
    """Find simple cycles in a directed graph.

    Parameters
    ----------
    graph
        Directed graph.
    limit
        Maximum number of cycles to return (None for all).

    Returns
    -------
    list[list[Any]]
        List of cycles as node lists.
    """
    return simple_cycles_by_id(graph, limit=limit)


def topological_layers(graph: GraphInput) -> dict[Any, int]:
    """Compute topological layer for each node in a DAG.

    Parameters
    ----------
    graph
        Directed acyclic graph.

    Returns
    -------
    dict[Any, int]
        Node to layer mapping (0 for roots).

    Notes
    -----
    If the graph contains cycles, rustworkx raises ``DAGHasCycle``.
    """
    return topological_layers_by_id(graph)


def condensation_layers(
    graph: GraphInput,
    scc_result: SCCResult,
) -> dict[Any, int]:
    """Compute layers based on SCC condensation.

    Parameters
    ----------
    graph
        Original directed graph.
    scc_result
        SCC analysis result with condensation.

    Returns
    -------
    dict[Any, int]
        Node to layer mapping based on condensation.
    """
    if scc_result.condensation is None:
        return {}
    comp_layers = topological_layers(scc_result.condensation)
    store = ensure_store(graph)
    return {
        node_id: comp_layers.get(scc_result.node_to_component.get(node_id, -1), 0)
        for node_id in store.node_ids()
    }


def component_metadata(graph: GraphInput) -> ComponentBundle:
    """Return weak component, SCC, cycle, and layer metadata.

    Returns
    -------
    ComponentBundle
        Component metrics for each node in the graph.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return ComponentBundle(
            component_id={},
            component_size={},
            scc_id={},
            scc_size={},
            in_cycle={},
            layer={},
        )

    weak_infos = find_weakly_connected(store)
    component_id: dict[Any, int] = {}
    component_size: dict[Any, int] = {}
    for info in weak_infos:
        for node in info.nodes:
            component_id[node] = info.component_id
            component_size[node] = info.size

    scc_result = find_strongly_connected(store, compute_condensation=True)
    scc_id: dict[Any, int] = scc_result.node_to_component
    scc_size: dict[Any, int] = {}
    for comp in scc_result.components:
        for node in comp.nodes:
            scc_size[node] = comp.size
    in_cycle = {node: scc_size.get(node, 1) > 1 for node in store.node_ids()}

    layer_map: dict[Any, int] = {}
    if scc_result.condensation is not None:
        condensation_layer = topological_layers(scc_result.condensation)
        layer_map = {
            node: condensation_layer.get(scc_id.get(node, 0), 0) for node in store.node_ids()
        }

    return ComponentBundle(
        component_id=component_id,
        component_size=component_size,
        scc_id=scc_id,
        scc_size=scc_size,
        in_cycle=in_cycle,
        layer=layer_map,
    )


def component_ids_undirected(graph: GraphInput) -> tuple[dict[Any, int], dict[Any, int]]:
    """Return component ids and sizes for undirected graphs.

    Returns
    -------
    tuple[dict[Any, int], dict[Any, int]]
        Component ids and component sizes keyed by node.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}, {}

    comp_infos = find_connected(store)
    component_id: dict[Any, int] = {}
    component_size: dict[Any, int] = {}
    for info in comp_infos:
        for node in info.nodes:
            component_id[node] = info.component_id
            component_size[node] = info.size
    return component_id, component_size


def _component_layers(graph: GraphInput) -> int | None:
    store = ensure_store(graph)
    if not store.is_directed:
        return None
    return compute_condensation_layer_count(store)


def _diameter_and_spl(graph: GraphInput) -> tuple[float | None, float | None]:
    diameter = compute_diameter_estimate(graph)
    avg_spl = compute_avg_shortest_path_length(graph)
    return diameter, avg_spl


def global_graph_stats(graph: GraphInput) -> GlobalGraphStats:
    """Return global statistics for the provided graph.

    Returns
    -------
    GlobalGraphStats
        Global graph summary statistics.
    """
    diameter_estimate, avg_spl_estimate = _diameter_and_spl(graph)
    component_layers = _component_layers(graph)

    clustering_map = compute_clustering_coefficient(graph)
    avg_clustering = sum(clustering_map.values()) / len(clustering_map) if clustering_map else 0.0

    store = ensure_store(graph)
    if store.is_directed:
        weak_infos = find_weakly_connected(store)
        weak_component_count = len(weak_infos)
        scc_result = find_strongly_connected(store)
        scc_count = len(scc_result.components)
    else:
        conn_infos = find_connected(store)
        weak_component_count = len(conn_infos)
        scc_count = weak_component_count

    return GlobalGraphStats(
        node_count=store.graph.num_nodes(),
        edge_count=store.graph.num_edges(),
        weak_component_count=weak_component_count,
        scc_count=scc_count,
        component_layers=component_layers,
        avg_clustering=avg_clustering,
        diameter_estimate=diameter_estimate,
        avg_shortest_path_estimate=avg_spl_estimate,
    )


__all__ = [
    "ComponentInfo",
    "ComponentStats",
    "SCCResult",
    "component_ids_undirected",
    "component_metadata",
    "compute_component_stats",
    "condensation_layers",
    "find_articulation_points",
    "find_bridges",
    "find_connected",
    "find_cycles",
    "find_strongly_connected",
    "find_weakly_connected",
    "global_graph_stats",
    "topological_layers",
]
