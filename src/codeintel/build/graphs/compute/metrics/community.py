"""Pure community detection computation functions."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import rustworkx as rx

from codeintel.build.graphs.rx.algos import (
    GraphInput,
    bridges_by_id,
    connected_components_by_id,
    ensure_store,
    to_undirected_store,
)
from codeintel.build.graphs.rx.iterators import (
    edge_weight_map,
    iter_edge_payloads,
    neighbors_by_index,
)
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.store import RxGraphStore


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _neighbor_map(store: RxGraphStore) -> dict[int, list[int]]:
    return neighbors_by_index(store, include_self=False)


def _component_sort_key(component: set[Any]) -> tuple[int, tuple[str, str]]:
    if not component:
        return (0, ("", ""))
    smallest = min(component, key=stable_key)
    return (len(component), stable_key(smallest))


def _component_size_without_edge(
    start: int,
    neighbors: dict[int, list[int]],
    blocked: tuple[int, int],
) -> int:
    blocked_key = _edge_key(*blocked)
    visited: set[int] = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in neighbors.get(current, []):
            if _edge_key(current, neighbor) == blocked_key:
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            stack.append(neighbor)
    return len(visited)


def _store_without_edges(
    store: RxGraphStore,
    removed_edges: set[tuple[int, int]],
) -> RxGraphStore:
    edge_graph = rx.PyGraph(multigraph=False)
    edge_nodes: dict[int, object] = {}
    for src_idx, dst_idx, _payload in iter_edge_payloads(store):
        if _edge_key(src_idx, dst_idx) in removed_edges:
            continue
        edge_nodes[src_idx] = store.graph.get_node_data(src_idx)
        edge_nodes[dst_idx] = store.graph.get_node_data(dst_idx)
    idx_map: dict[int, int] = {}
    ordered_edge_nodes = sorted(
        edge_nodes,
        key=lambda idx: stable_key(store.index_to_id.get(idx, idx)),
    )
    for old_idx in ordered_edge_nodes:
        payload = edge_nodes[old_idx]
        idx_map[old_idx] = edge_graph.add_node(payload)
    for src_idx, dst_idx, payload in iter_edge_payloads(store):
        if _edge_key(src_idx, dst_idx) in removed_edges:
            continue
        src_new = idx_map.get(src_idx)
        dst_new = idx_map.get(dst_idx)
        if src_new is None or dst_new is None:
            continue
        edge_graph.add_edge(src_new, dst_new, payload)
    isolate_graph = rx.PyGraph(multigraph=False)
    isolate_indices = sorted(
        store.graph.node_indices(),
        key=lambda idx: stable_key(store.index_to_id.get(idx, idx)),
    )
    isolate_graph.add_nodes_from([store.graph.get_node_data(idx) for idx in isolate_indices])
    merged = rx.union(edge_graph, isolate_graph, merge_nodes=True, merge_edges=True)
    return RxGraphStore.from_rx_graph(
        merged,
        weight_policy=store.weight_policy,
        numeric_policy=store.numeric_policy,
    )


def _assign_communities(
    components: list[set[Any]],
    *,
    sort_components_flag: bool,
) -> dict[Any, int]:
    ordered = (
        sorted(components, key=_component_sort_key) if sort_components_flag else list(components)
    )
    mapping: dict[Any, int] = {}
    for community_id, component in enumerate(ordered):
        for node_id in component:
            mapping[node_id] = community_id
    return mapping


def _bridge_split_components(
    store: RxGraphStore,
    *,
    min_component_size: int,
) -> tuple[list[set[Any]], RxGraphStore]:
    neighbors = _neighbor_map(store)
    removed_edges: set[tuple[int, int]] = set()
    total_nodes = store.graph.num_nodes()
    for left_id, right_id in bridges_by_id(store):
        src_idx = store.id_to_index.get(left_id)
        dst_idx = store.id_to_index.get(right_id)
        if src_idx is None or dst_idx is None:
            continue
        size_left = _component_size_without_edge(src_idx, neighbors, (src_idx, dst_idx))
        size_right = total_nodes - size_left
        if size_left >= min_component_size and size_right >= min_component_size:
            removed_edges.add(_edge_key(src_idx, dst_idx))
    merged_store = _store_without_edges(store, removed_edges)
    if merged_store.graph.num_nodes() == 0:
        return [], merged_store
    components = connected_components_by_id(merged_store)
    return components, merged_store


def _detect_communities_bridge_split(
    graph: GraphInput,
    *,
    min_component_size: int,
    weight: str | None,
    resolution: float,
    seed: int | None,
) -> dict[Any, int]:
    store = ensure_store(graph, weight=weight)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return {}
    adjusted_min_size = min_component_size if resolution >= 1.0 else max(1, min_component_size - 1)
    components, _component_store = _bridge_split_components(
        work_store,
        min_component_size=adjusted_min_size,
    )
    if not components:
        return {}
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(components)
    return _assign_communities(components, sort_components_flag=seed is None)


def detect_communities_bridge_split(
    graph: GraphInput,
    *,
    weight: str | None = None,
    resolution: float = 1.0,
    seed: int | None = None,
) -> dict[Any, int]:
    """Detect communities using a deterministic bridge-split heuristic.

    Returns
    -------
    dict[Any, int]
        Community assignments keyed by node identifier.
    """
    return _detect_communities_bridge_split(
        graph,
        min_component_size=2,
        weight=weight,
        resolution=resolution,
        seed=seed,
    )


def detect_communities_greedy(
    graph: GraphInput,
    *,
    weight: str | None = None,
    resolution: float = 1.0,
) -> dict[Any, int]:
    """Detect communities using greedy modularity-style splitting.

    Returns
    -------
    dict[Any, int]
        Community assignments keyed by node identifier.
    """
    return detect_communities_bridge_split(
        graph,
        weight=weight,
        resolution=resolution,
        seed=None,
    )


def detect_communities_louvain(
    graph: GraphInput,
    *,
    weight: str | None = None,
    resolution: float = 1.0,
    seed: int | None = None,
) -> dict[Any, int]:
    """Detect communities using the bridge-split heuristic (legacy name).

    Returns
    -------
    dict[Any, int]
        Community assignments keyed by node identifier.
    """
    return detect_communities_bridge_split(
        graph,
        weight=weight,
        resolution=resolution,
        seed=seed,
    )


def detect_communities_label_propagation(
    graph: GraphInput,
) -> dict[Any, int]:
    """Detect communities using the bridge-split heuristic (legacy name).

    Returns
    -------
    dict[Any, int]
        Community assignments keyed by node identifier.
    """
    return detect_communities_bridge_split(
        graph,
        weight=None,
        resolution=1.0,
        seed=None,
    )


def compute_modularity(
    graph: GraphInput,
    communities: dict[Any, int],
    *,
    weight: str | None = None,
    resolution: float = 1.0,
) -> float:
    """Compute modularity of a community partition.

    Returns
    -------
    float
        Modularity score for the provided community partition.
    """
    store = ensure_store(graph, weight=weight)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0 or not communities:
        return 0.0
    node_to_comm = _node_communities(work_store, communities)
    if not node_to_comm:
        return 0.0
    edge_weights = edge_weight_map(work_store)
    total_weight = sum(edge_weights.values())
    if total_weight == 0.0:
        return 0.0
    degree, intra = _community_weights(edge_weights, node_to_comm)
    return _modularity_score(
        total_weight=total_weight,
        degree=degree,
        intra=intra,
        node_to_comm=node_to_comm,
        resolution=resolution,
    )


def _node_communities(
    store: RxGraphStore,
    communities: Mapping[Any, int],
) -> dict[int, int]:
    node_to_comm: dict[int, int] = {}
    for node_id, community_id in communities.items():
        node_idx = store.id_to_index.get(node_id)
        if node_idx is None:
            continue
        node_to_comm[node_idx] = community_id
    return node_to_comm


def _community_weights(
    edge_weights: Mapping[tuple[int, int], float],
    node_to_comm: Mapping[int, int],
) -> tuple[dict[int, float], dict[int, float]]:
    degree: dict[int, float] = dict.fromkeys(node_to_comm, 0.0)
    intra: dict[int, float] = {}
    for (src_idx, dst_idx), weight_val in edge_weights.items():
        if src_idx == dst_idx:
            degree[src_idx] = degree.get(src_idx, 0.0) + weight_val * 2.0
        else:
            degree[src_idx] = degree.get(src_idx, 0.0) + weight_val
            degree[dst_idx] = degree.get(dst_idx, 0.0) + weight_val
        comm_src = node_to_comm.get(src_idx)
        comm_dst = node_to_comm.get(dst_idx)
        if comm_src is not None and comm_src == comm_dst:
            intra[comm_src] = intra.get(comm_src, 0.0) + weight_val
    return degree, intra


def _modularity_score(
    *,
    total_weight: float,
    degree: Mapping[int, float],
    intra: Mapping[int, float],
    node_to_comm: Mapping[int, int],
    resolution: float,
) -> float:
    modularity = 0.0
    for comm_id in set(node_to_comm.values()):
        degree_sum = sum(
            degree.get(idx, 0.0) for idx, node_comm in node_to_comm.items() if node_comm == comm_id
        )
        modularity += (intra.get(comm_id, 0.0) / total_weight) - resolution * (
            (degree_sum / (2.0 * total_weight)) ** 2
        )
    return float(modularity)


__all__ = [
    "compute_modularity",
    "detect_communities_bridge_split",
    "detect_communities_greedy",
    "detect_communities_label_propagation",
    "detect_communities_louvain",
]
