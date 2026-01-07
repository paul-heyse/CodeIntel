"""Pure community detection computation functions."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any, cast

import rustworkx as rx

from codeintel.build.graphs.rx.algos import GraphInput, ensure_store, to_undirected_store
from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, stable_key
from codeintel.build.graphs.rx.store import RxGraphStore


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _neighbor_map(store: RxGraphStore) -> dict[int, list[int]]:
    neighbors: dict[int, set[int]] = {idx: set() for idx in store.graph.node_indices()}
    for src_idx, dst_idx in store.graph.edge_list():
        if src_idx == dst_idx:
            continue
        neighbors[src_idx].add(dst_idx)
        neighbors[dst_idx].add(src_idx)
    return {
        idx: sorted(values, key=lambda node_idx: stable_key(store.index_to_id[node_idx]))
        for idx, values in neighbors.items()
    }


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


def _components_without_edges(
    neighbors: dict[int, list[int]],
    node_order: Iterable[int],
    removed_edges: set[tuple[int, int]],
) -> list[set[int]]:
    visited: set[int] = set()
    components: list[set[int]] = []
    for node_idx in node_order:
        if node_idx in visited:
            continue
        component: set[int] = set()
        stack = [node_idx]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in neighbors.get(current, []):
                if _edge_key(current, neighbor) in removed_edges:
                    continue
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components


def _component_sort_key(store: RxGraphStore, component: set[int]) -> tuple[str, str]:
    if not component:
        return ("", "")
    smallest = min(
        (store.index_to_id[idx] for idx in component),
        key=stable_key,
    )
    return stable_key(smallest)


def _assign_communities(
    store: RxGraphStore,
    components: list[set[int]],
    *,
    sort_components: bool,
) -> dict[Any, int]:
    ordered = (
        sorted(components, key=lambda comp: _component_sort_key(store, comp))
        if sort_components
        else list(components)
    )
    mapping: dict[Any, int] = {}
    for community_id, component in enumerate(ordered):
        for node_idx in component:
            mapping[store.index_to_id[node_idx]] = community_id
    return mapping


def _bridge_split_components(
    store: RxGraphStore,
    *,
    min_component_size: int,
) -> list[set[int]]:
    neighbors = _neighbor_map(store)
    node_order = sorted(neighbors.keys(), key=lambda idx: stable_key(store.index_to_id[idx]))
    removed_edges: set[tuple[int, int]] = set()
    total_nodes = len(node_order)
    undirected_graph = cast("rx.PyGraph", store.graph)
    for edge in rx.bridges(undirected_graph):
        src_idx, dst_idx = cast("tuple[int, int]", edge)
        size_left = _component_size_without_edge(src_idx, neighbors, (src_idx, dst_idx))
        size_right = total_nodes - size_left
        if size_left >= min_component_size and size_right >= min_component_size:
            removed_edges.add(_edge_key(src_idx, dst_idx))
    return _components_without_edges(neighbors, node_order, removed_edges)


def _detect_communities(
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
    components = _bridge_split_components(work_store, min_component_size=adjusted_min_size)
    if not components:
        return {}
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(components)
    return _assign_communities(work_store, components, sort_components=seed is None)


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
    return _detect_communities(
        graph,
        min_component_size=2,
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
    """Detect communities using a deterministic Louvain-style heuristic.

    Returns
    -------
    dict[Any, int]
        Community assignments keyed by node identifier.
    """
    return _detect_communities(
        graph,
        min_component_size=2,
        weight=weight,
        resolution=resolution,
        seed=seed,
    )


def detect_communities_label_propagation(
    graph: GraphInput,
) -> dict[Any, int]:
    """Detect communities using a deterministic label propagation heuristic.

    Returns
    -------
    dict[Any, int]
        Community assignments keyed by node identifier.
    """
    return _detect_communities(
        graph,
        min_component_size=2,
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
    edge_weights = _edge_weights(work_store)
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


def _edge_weights(store: RxGraphStore) -> dict[tuple[int, int], float]:
    edge_weights: dict[tuple[int, int], float] = {}
    for (src_idx, dst_idx), payload in zip(
        store.graph.edge_list(),
        store.graph.edges(),
        strict=True,
    ):
        edge_weights[_edge_key(src_idx, dst_idx)] = edge_weight_from_payload(payload)
    return edge_weights


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
    "detect_communities_greedy",
    "detect_communities_label_propagation",
    "detect_communities_louvain",
]
