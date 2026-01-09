"""Shared condensation helpers for rustworkx graphs."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import cast

import rustworkx as rx

from codeintel.build.graphs.rx.components import component_membership_by_id, sort_components
from codeintel.build.graphs.rx.store import RxGraphStore


def condensation_store(
    store: RxGraphStore,
    *,
    components: list[set[int]] | None = None,
    stable: bool = True,
    count_intercomponent_edges: bool = True,
) -> tuple[RxGraphStore, dict[Hashable, int]]:
    """Return a condensation store and node-id membership mapping.

    Parameters
    ----------
    store
        Input rustworkx graph store.
    components
        Optional precomputed components expressed as node-index sets.
    stable
        Whether to enforce stable component ordering.
    count_intercomponent_edges
        Whether to count inter-component edges as weights.

    Returns
    -------
    tuple[RxGraphStore, dict[Hashable, int]]
        Condensed graph store plus node-id -> component-id mapping.
    """
    if store.graph.num_nodes() == 0:
        return _empty_condensation(store)
    resolved_components = _resolve_components(store, components)
    sorted_components = _sorted_components(store, resolved_components, stable=stable)
    membership_by_id = component_membership_by_id(store, sorted_components)
    condensed = _build_condensed_store(store, component_count=len(sorted_components))
    if not count_intercomponent_edges:
        return condensed, membership_by_id
    edge_weights = _intercomponent_edge_weights(store, membership_by_id)
    _populate_intercomponent_edges(condensed, edge_weights)
    return condensed, membership_by_id


def _empty_condensation(store: RxGraphStore) -> tuple[RxGraphStore, dict[Hashable, int]]:
    empty_store = (
        RxGraphStore.directed(weight_policy=store.weight_policy)
        if store.is_directed
        else RxGraphStore.undirected(weight_policy=store.weight_policy)
    )
    return empty_store, {}


def _components_from_node_map(node_map: Sequence[object]) -> list[set[int]]:
    max_id = -1
    for comp_id in node_map:
        if isinstance(comp_id, int):
            max_id = max(max_id, comp_id)
    if max_id < 0:
        return []
    components: list[set[int]] = [set() for _ in range(max_id + 1)]
    for node_idx, comp_id in enumerate(node_map):
        if isinstance(comp_id, int):
            components[comp_id].add(node_idx)
    return components


def _resolve_components(
    store: RxGraphStore,
    components: list[set[int]] | None,
) -> list[set[int]]:
    if components is not None:
        return list(components)
    condensed = rx.condensation(store.graph)
    node_map = condensed.attrs.get("node_map")
    if isinstance(node_map, Sequence) and not isinstance(
        node_map,
        (str, bytes, bytearray, memoryview),
    ):
        resolved = _components_from_node_map(node_map)
        if resolved:
            return resolved
    if store.is_directed:
        directed_graph = cast("rx.PyDiGraph[object, float]", store.graph)
        return [set(comp) for comp in rx.strongly_connected_components(directed_graph)]
    undirected_graph = cast("rx.PyGraph[object, float]", store.graph)
    return [set(comp) for comp in rx.connected_components(undirected_graph)]


def _sorted_components(
    store: RxGraphStore,
    components: list[set[int]],
    *,
    stable: bool,
) -> list[set[int]]:
    return sort_components(store, components) if stable else list(components)


def _build_condensed_store(store: RxGraphStore, *, component_count: int) -> RxGraphStore:
    condensed = (
        RxGraphStore.directed(
            node_hint=component_count,
            weight_policy=store.weight_policy,
        )
        if store.is_directed
        else RxGraphStore.undirected(
            node_hint=component_count,
            weight_policy=store.weight_policy,
        )
    )
    for comp_id in range(component_count):
        condensed.ensure_node(comp_id)
    return condensed


def _intercomponent_edge_weights(
    store: RxGraphStore,
    membership_by_id: dict[Hashable, int],
) -> dict[tuple[int, int], float]:
    edge_weights: dict[tuple[int, int], float] = {}
    for src_idx, dst_idx in store.graph.edge_list():
        src_id = store.index_to_id.get(src_idx)
        dst_id = store.index_to_id.get(dst_idx)
        if src_id is None or dst_id is None:
            continue
        src_comp = membership_by_id.get(src_id)
        dst_comp = membership_by_id.get(dst_id)
        if src_comp is None or dst_comp is None or src_comp == dst_comp:
            continue
        key = _edge_weight_key(src_comp, dst_comp, directed=store.is_directed)
        edge_weights[key] = edge_weights.get(key, 0.0) + 1.0
    return edge_weights


def _edge_weight_key(
    src_comp: int,
    dst_comp: int,
    *,
    directed: bool,
) -> tuple[int, int]:
    if directed or src_comp <= dst_comp:
        return (src_comp, dst_comp)
    return (dst_comp, src_comp)


def _populate_intercomponent_edges(
    condensed: RxGraphStore,
    edge_weights: dict[tuple[int, int], float],
) -> None:
    for (src_comp, dst_comp), weight in edge_weights.items():
        condensed.add_weighted_edge(src_comp, dst_comp, weight=weight)


__all__ = ["condensation_store"]
