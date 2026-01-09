"""Pure component analysis functions.

This module provides stateless functions for computing graph components
and structural properties without any database or file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import rustworkx as rx

from codeintel.build.graphs.compute.metrics.statistics import (
    compute_avg_shortest_path_length,
    compute_condensation_layer_count,
    compute_diameter_estimate,
)
from codeintel.build.graphs.compute.metrics.structural import compute_clustering_coefficient
from codeintel.build.graphs.compute.metrics.types import ComponentBundle, GlobalGraphStats
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    ensure_directed_store,
    ensure_store,
    to_undirected_store,
)
from codeintel.build.graphs.rx.components import (
    component_membership_by_id,
    sort_components,
)
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, sorted_mapping, stable_key
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


def _directed_graph(store: RxGraphStore) -> rx.PyDiGraph:
    if not store.is_directed:
        message = "Expected a directed graph store"
        raise ValueError(message)
    return cast("rx.PyDiGraph", store.graph)


def _undirected_graph(store: RxGraphStore) -> rx.PyGraph:
    if store.is_directed:
        message = "Expected an undirected graph store"
        raise ValueError(message)
    return cast("rx.PyGraph", store.graph)


def _condensation_components(
    store: RxGraphStore,
) -> tuple[rx.PyDiGraph, list[set[int]], dict[int, int]] | None:
    condensed = cast("rx.PyDiGraph", rx.condensation(_directed_graph(store)))
    node_map = condensed.attrs.get("node_map")
    if not isinstance(node_map, Sequence) or isinstance(
        node_map, (str, bytes, bytearray, memoryview)
    ):
        return None
    components_by_id: dict[int, set[int]] = {}
    for node_idx, comp_id in enumerate(node_map):
        if isinstance(comp_id, int):
            components_by_id.setdefault(comp_id, set()).add(node_idx)
    if not components_by_id:
        return condensed, [], {}
    sorted_components = sort_components(store, components_by_id.values())
    component_lookup = {
        frozenset(component): old_id for old_id, component in components_by_id.items()
    }
    old_to_new: dict[int, int] = {}
    for new_id, component in enumerate(sorted_components):
        old_id = component_lookup.get(frozenset(component))
        if old_id is not None:
            old_to_new[old_id] = new_id
    return condensed, sorted_components, old_to_new


def _condensation_store(
    store: RxGraphStore,
    *,
    condensed_graph: rx.PyDiGraph,
    old_to_new: dict[int, int],
    component_count: int,
) -> RxGraphStore:
    if component_count == 0:
        return RxGraphStore.directed(
            weight_policy=store.weight_policy,
            numeric_policy=store.numeric_policy,
        )
    condensed_store = RxGraphStore.directed(
        node_hint=component_count,
        weight_policy=store.weight_policy,
        numeric_policy=store.numeric_policy,
    )
    for comp_id in range(component_count):
        condensed_store.ensure_node(comp_id)
    for src_idx, dst_idx in condensed_graph.edge_list():
        src_new = old_to_new.get(src_idx)
        dst_new = old_to_new.get(dst_idx)
        if src_new is None or dst_new is None or src_new == dst_new:
            continue
        payload = condensed_graph.get_edge_data(src_idx, dst_idx)
        weight = edge_weight_from_payload(payload)
        condensed_store.add_weighted_edge(src_new, dst_new, weight=weight)
    return condensed_store


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

    condensation_data = _condensation_components(store)
    if condensation_data is None:
        return SCCResult(components=(), node_to_component={})
    condensed_graph, sorted_sccs, old_to_new = condensation_data
    if not sorted_sccs:
        return SCCResult(components=(), node_to_component={})
    components: list[ComponentInfo] = []
    node_to_component = component_membership_by_id(store, sorted_sccs)
    for comp_id, comp in enumerate(sorted_sccs):
        nodes_frozen = frozenset(store.index_to_id[idx] for idx in comp)
        components.append(
            ComponentInfo(
                component_id=comp_id,
                size=len(comp),
                nodes=nodes_frozen,
            )
        )
    condensation = None
    if compute_condensation:
        condensation = _condensation_store(
            store,
            condensed_graph=condensed_graph,
            old_to_new=old_to_new,
            component_count=len(sorted_sccs),
        )

    return SCCResult(
        components=tuple(components),
        node_to_component=node_to_component,
        condensation=condensation,
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
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return []

    if store.is_directed:
        directed_graph = _directed_graph(store)
        components = [set(comp) for comp in rx.weakly_connected_components(directed_graph)]
    else:
        undirected_graph = _undirected_graph(store)
        components = [set(comp) for comp in rx.connected_components(undirected_graph)]
    sorted_components = sort_components(store, components)
    return [
        ComponentInfo(
            component_id=idx,
            size=len(comp),
            nodes=frozenset(store.index_to_id[node_idx] for node_idx in comp),
        )
        for idx, comp in enumerate(sorted_components)
    ]


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
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []

    undirected_graph = _undirected_graph(work_store)
    components = [set(comp) for comp in rx.connected_components(undirected_graph)]
    sorted_components = sort_components(work_store, components)
    return [
        ComponentInfo(
            component_id=idx,
            size=len(comp),
            nodes=frozenset(work_store.index_to_id[node_idx] for node_idx in comp),
        )
        for idx, comp in enumerate(sorted_components)
    ]


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
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []
    undirected_graph = _undirected_graph(work_store)
    bridges = []
    for edge in rx.bridges(undirected_graph):
        src_idx, dst_idx = cast("tuple[int, int]", edge)
        bridges.append((work_store.index_to_id[src_idx], work_store.index_to_id[dst_idx]))
    return sorted(bridges, key=stable_key)


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
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []
    undirected_graph = _undirected_graph(work_store)
    points = [work_store.index_to_id[idx] for idx in rx.articulation_points(undirected_graph)]
    return sorted(points, key=stable_key)


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
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []

    directed_graph = _directed_graph(store)
    cycles: list[list[Any]] = []
    for cycle in rx.simple_cycles(directed_graph):
        cycles.append([store.index_to_id[idx] for idx in cycle])
        if limit is not None and len(cycles) >= limit:
            break
    return cycles


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
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    directed_graph = _directed_graph(store)
    layer_map: dict[Any, int] = {}
    for layer, generation in enumerate(rx.topological_generations(directed_graph)):
        ordered = sorted(
            generation,
            key=lambda idx: stable_key(store.index_to_id[idx]),
        )
        for node_idx in ordered:
            layer_map[store.index_to_id[node_idx]] = layer
    return sorted_mapping(layer_map)


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
