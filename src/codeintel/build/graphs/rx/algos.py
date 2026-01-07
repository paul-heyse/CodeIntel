"""Rustworkx-first algorithm wrappers with deterministic outputs."""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import rustworkx as rx

from codeintel.build.graphs.rx.convert import store_from_rx
from codeintel.build.graphs.rx.normalize import (
    NanPolicy,
    edge_weight_from_payload,
    normalize_mapping,
    sorted_mapping,
    stable_key,
)
from codeintel.build.graphs.rx.policies import GraphNumericPolicy
from codeintel.build.graphs.rx.store import RxGraphStore

RxGraph = rx.PyGraph | rx.PyDiGraph
DirectedRxGraph = rx.PyDiGraph
UndirectedRxGraph = rx.PyGraph
GraphInput = RxGraphStore | RxGraph

_MIN_BETWEENNESS_NODES = 2
_MIN_CLUSTERING_DEGREE = 2


def _apply_tolerance(value: float, *, abs_tol: float, rel_tol: float) -> float:
    if abs_tol == 0.0 and rel_tol == 0.0:
        return value
    for target in (0.0, 1.0):
        if math.isclose(value, target, abs_tol=abs_tol, rel_tol=rel_tol):
            return target
    return value


def _resolve_nan_policy(store: RxGraphStore, nan_policy: NanPolicy | None) -> NanPolicy:
    if nan_policy is None:
        return store.numeric_policy.nan_policy
    return nan_policy


def _normalize_float_mapping(
    mapping: Mapping[Hashable, float],
    *,
    nan_policy: NanPolicy,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> dict[Hashable, float]:
    normalized = normalize_mapping(mapping, nan_policy=nan_policy)
    if abs_tol == 0.0 and rel_tol == 0.0:
        return normalized
    return {
        key: _apply_tolerance(value, abs_tol=abs_tol, rel_tol=rel_tol)
        for key, value in normalized.items()
    }


def _sorted_node_indices(store: RxGraphStore) -> list[int]:
    return [store.id_to_index[node_id] for node_id in store.node_ids()]


def _sorted_neighbors(store: RxGraphStore, nodes: Iterable[int]) -> list[int]:
    return sorted(nodes, key=lambda idx: stable_key(store.index_to_id[idx]))


def _edge_weight_fn(*, nan_policy: NanPolicy) -> Callable[[object], float]:
    def weight_fn(payload: object) -> float:
        return edge_weight_from_payload(payload, nan_policy=nan_policy)

    return weight_fn


def _constant_weight_fn(_payload: object) -> float:
    return 1.0


@dataclass(frozen=True, slots=True)
class PagerankOptions:
    """Options for PageRank computation."""

    alpha: float = 0.85
    max_iter: int = 100
    tol: float = 1e-6
    weight: str | None = None
    nan_policy: NanPolicy | None = None


@dataclass(frozen=True, slots=True)
class BetweennessOptions:
    """Options for betweenness centrality computation."""

    normalized: bool = True
    k: int | None = None
    weight: str | None = None
    seed: int | None = None
    nan_policy: NanPolicy | None = None


def ensure_store(
    graph: GraphInput,
    *,
    weight: str | None = "weight",
    nan_policy: NanPolicy | None = None,
) -> RxGraphStore:
    """Coerce supported graph inputs into an RxGraphStore.

    Returns
    -------
    RxGraphStore
        Store wrapping the provided graph input.

    Raises
    ------
    TypeError
        If the input graph type is unsupported.
    """
    if isinstance(graph, RxGraphStore):
        return graph
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        return store_from_rx(graph)
    message = (
        f"Unsupported graph input: {type(graph).__name__} "
        f"(weight={weight}, nan_policy={nan_policy})"
    )
    raise TypeError(message)


def graph_to_store(graph: GraphInput) -> RxGraphStore:
    """Coerce a graph input into an RxGraphStore.

    Returns
    -------
    RxGraphStore
        Store representation of the input.
    """
    return ensure_store(graph)


def _directed_graph(store: RxGraphStore) -> DirectedRxGraph:
    if not store.is_directed:
        message = "Expected a directed graph store"
        raise ValueError(message)
    return cast("DirectedRxGraph", store.graph)


def _undirected_graph(store: RxGraphStore) -> UndirectedRxGraph:
    if store.is_directed:
        message = "Expected an undirected graph store"
        raise ValueError(message)
    return cast("UndirectedRxGraph", store.graph)


def to_undirected_store(store: RxGraphStore) -> RxGraphStore:
    """Return an undirected store representation for a graph.

    Returns
    -------
    RxGraphStore
        Undirected store representation of the input graph.
    """
    if not store.is_directed:
        return store
    return store.as_undirected()


def to_directed_store(store: RxGraphStore) -> RxGraphStore:
    """Return a directed store representation for a graph.

    Returns
    -------
    RxGraphStore
        Directed store representation of the input graph.
    """
    if store.is_directed:
        return store
    return store.as_directed()


def ensure_directed_store(
    graph: GraphInput,
    *,
    weight: str | None = "weight",
    nan_policy: NanPolicy | None = None,
) -> RxGraphStore:
    """Coerce supported inputs into a directed RxGraphStore.

    Returns
    -------
    RxGraphStore
        Directed store representation of the input graph.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    return to_directed_store(store)


def graph_node_count(graph: GraphInput) -> int:
    """Return the node count for the provided graph.

    Returns
    -------
    int
        Number of nodes in the graph.
    """
    if isinstance(graph, RxGraphStore):
        return graph.graph.num_nodes()
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        return graph.num_nodes()
    return 0


def graph_edge_count(graph: GraphInput) -> int:
    """Return the edge count for the provided graph.

    Returns
    -------
    int
        Number of edges in the graph.
    """
    if isinstance(graph, RxGraphStore):
        return graph.graph.num_edges()
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        return graph.num_edges()
    return 0


def pagerank_by_id(
    graph: GraphInput,
    *,
    options: PagerankOptions | None = None,
) -> dict[Hashable, float]:
    """Compute PageRank scores keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        PageRank scores keyed by node identifier.
    """
    resolved = options or PagerankOptions()
    store = ensure_directed_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, resolved.nan_policy)
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    directed_graph = _directed_graph(store)
    weight_fn: Callable[[object], float] | None = None
    if resolved.weight is not None:
        weight_fn = _edge_weight_fn(nan_policy=resolved_nan_policy)
    raw = rx.pagerank(
        directed_graph,
        alpha=resolved.alpha,
        weight_fn=weight_fn,
        tol=resolved.tol,
        max_iter=resolved.max_iter,
    )
    mapped = {store.index_to_id[idx]: float(score) for idx, score in raw.items()}
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def eigenvector_centrality_by_id(
    graph: GraphInput,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute eigenvector centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Eigenvector centrality scores keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return {}
    undirected_graph = _undirected_graph(work_store)
    weight_fn: Callable[[object], float] | None = None
    if weight is not None:
        weight_fn = _edge_weight_fn(nan_policy=resolved_nan_policy)
    raw = rx.graph_eigenvector_centrality(
        undirected_graph,
        weight_fn=weight_fn,
        max_iter=max_iter,
        tol=tol,
    )
    mapped = {work_store.index_to_id[idx]: float(score) for idx, score in raw.items()}
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def closeness_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    wf_improved: bool = True,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute closeness centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Closeness centrality scores keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    if store.graph.num_nodes() == 0:
        return {}
    if weight is None:
        if store.is_directed:
            directed_graph = _directed_graph(store)
            raw = rx.digraph_closeness_centrality(directed_graph, wf_improved=wf_improved)
        else:
            undirected_graph = _undirected_graph(store)
            raw = rx.graph_closeness_centrality(undirected_graph, wf_improved=wf_improved)
    else:
        weight_fn = _edge_weight_fn(nan_policy=resolved_nan_policy)
        if store.is_directed:
            directed_graph = _directed_graph(store)
            raw = rx.digraph_newman_weighted_closeness_centrality(
                directed_graph,
                weight_fn=weight_fn,
                default_weight=1.0,
                wf_improved=wf_improved,
            )
        else:
            undirected_graph = _undirected_graph(store)
            raw = rx.graph_newman_weighted_closeness_centrality(
                undirected_graph,
                weight_fn=weight_fn,
                default_weight=1.0,
                wf_improved=wf_improved,
            )
    mapped = {store.index_to_id[idx]: float(score) for idx, score in raw.items()}
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def _edge_weight_map(store: RxGraphStore, *, nan_policy: NanPolicy) -> dict[tuple[int, int], float]:
    edge_map: dict[tuple[int, int], float] = {}
    for (src, dst), payload in zip(
        store.graph.edge_list(),
        store.graph.edges(),
        strict=True,
    ):
        key = (src, dst) if store.is_directed else (min(src, dst), max(src, dst))
        edge_map[key] = edge_weight_from_payload(payload, nan_policy=nan_policy)
    return edge_map


def _neighbor_map(
    store: RxGraphStore,
    *,
    include_self: bool = False,
) -> dict[int, list[int]]:
    neighbors: dict[int, set[int]] = {idx: set() for idx in store.graph.node_indices()}
    for src, dst in store.graph.edge_list():
        if src == dst:
            if include_self:
                neighbors[src].add(src)
            continue
        if store.is_directed:
            neighbors[src].add(dst)
        else:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return {idx: _sorted_neighbors(store, items) for idx, items in neighbors.items()}


def _weighted_neighbor_map(
    store: RxGraphStore,
    edge_weights: Mapping[tuple[int, int], float],
) -> dict[int, list[tuple[int, float]]]:
    neighbors: dict[int, list[tuple[int, float]]] = {idx: [] for idx in store.graph.node_indices()}
    for src, dst in store.graph.edge_list():
        key = (src, dst) if store.is_directed else (min(src, dst), max(src, dst))
        weight = edge_weights.get(key, 1.0)
        if src != dst:
            neighbors[src].append((dst, weight))
            if not store.is_directed:
                neighbors[dst].append((src, weight))
    for idx, items in neighbors.items():
        neighbors[idx] = sorted(items, key=lambda item: stable_key(store.index_to_id[item[0]]))
    return neighbors


def _brandes_unweighted(
    neighbors: Mapping[int, Sequence[int]],
    sources: Sequence[int],
) -> dict[int, float]:
    betweenness = dict.fromkeys(neighbors, 0.0)
    for source in sources:
        stack: list[int] = []
        predecessors: dict[int, list[int]] = {node: [] for node in neighbors}
        sigma = dict.fromkeys(neighbors, 0.0)
        distance = dict.fromkeys(neighbors, -1)
        sigma[source] = 1.0
        distance[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            stack.append(node)
            for neighbor in neighbors[node]:
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1
                if distance[neighbor] == distance[node] + 1:
                    sigma[neighbor] += sigma[node]
                    predecessors[neighbor].append(node)
        delta = dict.fromkeys(neighbors, 0.0)
        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                if sigma[node] != 0:
                    delta[pred] += (sigma[pred] / sigma[node]) * (1.0 + delta[node])
            if node != source:
                betweenness[node] += delta[node]
    return betweenness


def _brandes_weighted(
    neighbors: Mapping[int, Sequence[tuple[int, float]]],
    sources: Sequence[int],
    *,
    numeric_policy: GraphNumericPolicy,
) -> dict[int, float]:
    betweenness = dict.fromkeys(neighbors, 0.0)
    for source in sources:
        stack, predecessors, sigma = _brandes_weighted_paths(
            neighbors,
            source,
            numeric_policy=numeric_policy,
        )
        delta = _brandes_dependency(predecessors, sigma, stack)
        for node in delta:
            if node != source:
                betweenness[node] += delta[node]
    return betweenness


def _brandes_weighted_paths(
    neighbors: Mapping[int, Sequence[tuple[int, float]]],
    source: int,
    *,
    numeric_policy: GraphNumericPolicy,
) -> tuple[list[int], dict[int, list[int]], dict[int, float]]:
    stack: list[int] = []
    predecessors: dict[int, list[int]] = {node: [] for node in neighbors}
    sigma = dict.fromkeys(neighbors, 0.0)
    distance = dict.fromkeys(neighbors, math.inf)
    sigma[source] = 1.0
    distance[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        dist, node = heapq.heappop(queue)
        if dist > distance[node]:
            continue
        stack.append(node)
        for neighbor, weight in neighbors[node]:
            path_dist = distance[node] + weight
            if path_dist < distance[neighbor] - numeric_policy.dijkstra_abs_tol:
                distance[neighbor] = path_dist
                heapq.heappush(queue, (path_dist, neighbor))
                sigma[neighbor] = sigma[node]
                predecessors[neighbor] = [node]
            elif math.isclose(
                path_dist,
                distance[neighbor],
                rel_tol=numeric_policy.dijkstra_rel_tol,
                abs_tol=numeric_policy.dijkstra_abs_tol,
            ):
                sigma[neighbor] += sigma[node]
                predecessors[neighbor].append(node)
    return stack, predecessors, sigma


def _brandes_dependency(
    predecessors: Mapping[int, Sequence[int]],
    sigma: Mapping[int, float],
    stack: Sequence[int],
) -> dict[int, float]:
    delta = dict.fromkeys(predecessors, 0.0)
    for node in reversed(stack):
        for pred in predecessors[node]:
            if sigma[node] != 0:
                delta[pred] += (sigma[pred] / sigma[node]) * (1.0 + delta[node])
    return delta


def _rescale_betweenness(
    betweenness: dict[int, float],
    *,
    node_count: int,
    normalized: bool,
    directed: bool,
    sampled: int | None,
) -> dict[int, float]:
    if node_count <= _MIN_BETWEENNESS_NODES:
        return dict.fromkeys(betweenness, 0.0)
    scale = 1.0
    if sampled is not None and sampled > 0:
        scale *= node_count / sampled
    if normalized:
        if directed:
            scale *= 1.0 / ((node_count - 1) * (node_count - 2))
        else:
            scale *= 2.0 / ((node_count - 1) * (node_count - 2))
    elif not directed:
        scale *= 0.5
    if scale != 1.0:
        for node in betweenness:
            betweenness[node] *= scale
    return betweenness


def _betweenness_builtin_by_id(
    store: RxGraphStore,
    *,
    normalized: bool,
) -> dict[Hashable, float]:
    if store.is_directed:
        directed_graph = _directed_graph(store)
        raw = rx.digraph_betweenness_centrality(
            directed_graph,
            normalized=normalized,
        )
    else:
        undirected_graph = _undirected_graph(store)
        raw = rx.graph_betweenness_centrality(
            undirected_graph,
            normalized=normalized,
        )
    return {store.index_to_id[idx]: float(val) for idx, val in raw.items()}


def _resolve_sampled_indices(
    store: RxGraphStore,
    *,
    k: int | None,
    seed: int | None,
) -> tuple[list[int], int | None]:
    indices = _sorted_node_indices(store)
    if k is None:
        return indices, None
    if k < len(indices):
        rng = random.Random(seed)
        return rng.sample(indices, k), k
    return indices, len(indices)


def betweenness_by_id(
    graph: GraphInput,
    *,
    options: BetweennessOptions | None = None,
) -> dict[Hashable, float]:
    """Compute betweenness centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Betweenness centrality scores keyed by node identifier.
    """
    resolved = options or BetweennessOptions()
    store = ensure_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, resolved.nan_policy)
    numeric_policy = store.numeric_policy
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    if resolved.weight is None and resolved.k is None:
        mapped = _betweenness_builtin_by_id(store, normalized=resolved.normalized)
        return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)

    indices, sampled = _resolve_sampled_indices(store, k=resolved.k, seed=resolved.seed)

    if resolved.weight is None:
        neighbors = _neighbor_map(store, include_self=False)
        betweenness = _brandes_unweighted(neighbors, indices)
    else:
        edge_weights = _edge_weight_map(store, nan_policy=resolved_nan_policy)
        neighbors = _weighted_neighbor_map(store, edge_weights)
        betweenness = _brandes_weighted(neighbors, indices, numeric_policy=numeric_policy)

    rescaled = _rescale_betweenness(
        betweenness,
        node_count=node_count,
        normalized=resolved.normalized,
        directed=store.is_directed,
        sampled=sampled,
    )
    mapped = {store.index_to_id[idx]: float(val) for idx, val in rescaled.items()}
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def harmonic_centrality_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute harmonic centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Harmonic centrality scores keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    numeric_policy = store.numeric_policy
    if store.graph.num_nodes() == 0:
        return {}
    weight_fn: Callable[[object], float] = _constant_weight_fn
    if weight is not None:
        weight_fn = _edge_weight_fn(nan_policy=resolved_nan_policy)
    result: dict[Hashable, float] = {}
    directed_graph: DirectedRxGraph | None = None
    undirected_graph: UndirectedRxGraph | None = None
    for node_id in store.node_ids():
        node_idx = store.id_to_index[node_id]
        if store.is_directed:
            if directed_graph is None:
                directed_graph = _directed_graph(store)
            lengths = rx.digraph_dijkstra_shortest_path_lengths(
                directed_graph,
                node_idx,
                weight_fn,
            )
        else:
            if undirected_graph is None:
                undirected_graph = _undirected_graph(store)
            lengths = rx.graph_dijkstra_shortest_path_lengths(
                undirected_graph,
                node_idx,
                weight_fn,
            )
        if not lengths:
            result[node_id] = 0.0
            continue
        total = 0.0
        for distance in lengths.values():
            if distance > 0:
                total += 1.0 / float(distance)
        result[node_id] = total
    return _normalize_float_mapping(
        result,
        nan_policy=resolved_nan_policy,
        abs_tol=numeric_policy.harmonic_abs_tol,
        rel_tol=numeric_policy.harmonic_rel_tol,
    )


def degree_centrality_by_id(graph: GraphInput) -> dict[Hashable, float]:
    """Compute degree centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Degree centrality scores keyed by node identifier.
    """
    store = ensure_store(graph)
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    if node_count == 1:
        node_id = store.node_ids()[0]
        return {node_id: 1.0}
    result: dict[Hashable, float] = {}
    if store.is_directed:
        directed_graph = _directed_graph(store)
        in_raw = rx.in_degree_centrality(directed_graph)
        out_raw = rx.out_degree_centrality(directed_graph)
        for node_id in store.node_ids():
            idx = store.id_to_index[node_id]
            result[node_id] = float(in_raw.get(idx, 0.0)) + float(out_raw.get(idx, 0.0))
    else:
        undirected_graph = _undirected_graph(store)
        raw = rx.degree_centrality(undirected_graph)
        for node_id in store.node_ids():
            idx = store.id_to_index[node_id]
            result[node_id] = float(raw.get(idx, 0.0))
    return _normalize_float_mapping(result, nan_policy=store.numeric_policy.nan_policy)


def in_degree_centrality_by_id(graph: GraphInput) -> dict[Hashable, float]:
    """Compute in-degree centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        In-degree centrality scores keyed by node identifier.
    """
    store = ensure_store(graph)
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    if node_count == 1:
        node_id = store.node_ids()[0]
        return {node_id: 1.0}
    result: dict[Hashable, float] = {}
    directed_graph = _directed_graph(store)
    raw = rx.in_degree_centrality(directed_graph)
    for node_id in store.node_ids():
        idx = store.id_to_index[node_id]
        result[node_id] = float(raw.get(idx, 0.0))
    return _normalize_float_mapping(result, nan_policy=store.numeric_policy.nan_policy)


def out_degree_centrality_by_id(graph: GraphInput) -> dict[Hashable, float]:
    """Compute out-degree centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Out-degree centrality scores keyed by node identifier.
    """
    store = ensure_store(graph)
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    if node_count == 1:
        node_id = store.node_ids()[0]
        return {node_id: 1.0}
    result: dict[Hashable, float] = {}
    directed_graph = _directed_graph(store)
    raw = rx.out_degree_centrality(directed_graph)
    for node_id in store.node_ids():
        idx = store.id_to_index[node_id]
        result[node_id] = float(raw.get(idx, 0.0))
    return _normalize_float_mapping(result, nan_policy=store.numeric_policy.nan_policy)


def _triangle_counts(
    *,
    neighbor_map: Mapping[int, Sequence[int]],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node_idx, neighbors in neighbor_map.items():
        count = 0
        for i, left in enumerate(neighbors):
            left_neighbors = set(neighbor_map[left])
            for right in neighbors[i + 1 :]:
                if right in left_neighbors:
                    count += 1
        counts[node_idx] = count
    return counts


def triangles_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Compute triangle counts keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Triangle counts keyed by node identifier.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_map(work_store, include_self=False)
    counts = _triangle_counts(neighbor_map=neighbors)
    mapped = {work_store.index_to_id[idx]: int(val) for idx, val in counts.items()}
    return sorted_mapping(mapped)


def _edge_weight_lookup(
    edge_weights: Mapping[tuple[int, int], float],
    left: int,
    right: int,
) -> float:
    key = (left, right) if left <= right else (right, left)
    return edge_weights.get(key, 0.0)


def clustering_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute clustering coefficients keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Clustering coefficients keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    work_store = to_undirected_store(store)
    numeric_policy = work_store.numeric_policy
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_map(work_store, include_self=False)
    if weight is None:
        result = _clustering_unweighted(work_store, neighbors)
        return _normalize_float_mapping(result, nan_policy=resolved_nan_policy)
    result = _clustering_weighted(work_store, neighbors, nan_policy=resolved_nan_policy)
    return _normalize_float_mapping(
        result,
        nan_policy=resolved_nan_policy,
        abs_tol=numeric_policy.clustering_abs_tol,
        rel_tol=numeric_policy.clustering_rel_tol,
    )


def _clustering_unweighted(
    store: RxGraphStore,
    neighbors: Mapping[int, Sequence[int]],
) -> dict[Hashable, float]:
    triangles = _triangle_counts(neighbor_map=neighbors)
    result: dict[Hashable, float] = {}
    for node_idx, neighbor_list in neighbors.items():
        degree = len(neighbor_list)
        if degree < _MIN_CLUSTERING_DEGREE:
            result[store.index_to_id[node_idx]] = 0.0
            continue
        coeff = (2.0 * triangles.get(node_idx, 0)) / (degree * (degree - 1))
        result[store.index_to_id[node_idx]] = float(coeff)
    return result


def _clustering_weighted(
    store: RxGraphStore,
    neighbors: Mapping[int, Sequence[int]],
    *,
    nan_policy: NanPolicy,
) -> dict[Hashable, float]:
    edge_weights = _edge_weight_map(store, nan_policy=nan_policy)
    max_weight = max(edge_weights.values(), default=1.0)
    result: dict[Hashable, float] = {}
    for node_idx, neighbor_list in neighbors.items():
        degree = len(neighbor_list)
        if degree < _MIN_CLUSTERING_DEGREE:
            result[store.index_to_id[node_idx]] = 0.0
            continue
        weighted_triangles = _weighted_triangles(
            node_idx,
            neighbor_list,
            neighbors,
            edge_weights=edge_weights,
            max_weight=max_weight,
        )
        coeff = (2.0 * weighted_triangles) / (degree * (degree - 1))
        result[store.index_to_id[node_idx]] = coeff
    return result


def _weighted_triangles(
    node_idx: int,
    neighbors: Sequence[int],
    neighbor_map: Mapping[int, Sequence[int]],
    *,
    edge_weights: Mapping[tuple[int, int], float],
    max_weight: float,
) -> float:
    weighted_triangles = 0.0
    seen: set[int] = set()
    neighbor_set = set(neighbors)
    for neighbor in neighbors:
        seen.add(neighbor)
        neighbor_neighbors = set(neighbor_map.get(neighbor, [])) - seen
        weight_ij = _edge_weight_lookup(edge_weights, node_idx, neighbor) / max_weight
        for other in neighbor_neighbors & neighbor_set:
            weight_jk = _edge_weight_lookup(edge_weights, neighbor, other) / max_weight
            weight_ki = _edge_weight_lookup(edge_weights, other, node_idx) / max_weight
            product = weight_ij * weight_jk * weight_ki
            weighted_triangles += math.copysign(abs(product) ** (1.0 / 3.0), product)
    return weighted_triangles


def core_number_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Compute core numbers keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Core numbers keyed by node identifier.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return {}
    undirected_graph = _undirected_graph(work_store)
    raw = cast("dict[int, int]", rx.graph_core_number(undirected_graph))
    mapped = {work_store.index_to_id[idx]: int(val) for idx, val in raw.items()}
    return sorted_mapping(mapped)


def _neighbor_sets_with_self(store: RxGraphStore) -> dict[int, list[int]]:
    return _neighbor_map(store, include_self=True)


def _mutual_weight(
    edge_weights: Mapping[tuple[int, int], float],
    left: int,
    right: int,
) -> float:
    weight = _edge_weight_lookup(edge_weights, left, right)
    return weight * 2.0 if weight else 0.0


def _normalized_mutual_weight(
    edge_weights: Mapping[tuple[int, int], float],
    node: int,
    neighbor: int,
    *,
    scale: Mapping[int, float],
) -> float:
    denom = scale.get(node, 0.0)
    if denom == 0.0:
        return 0.0
    return _mutual_weight(edge_weights, node, neighbor) / denom


def _neighbor_scale(
    edge_weights: Mapping[tuple[int, int], float],
    neighbors: Mapping[int, Sequence[int]],
    norm: Callable[[Iterable[float]], float],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for node, node_neighbors in neighbors.items():
        weights = (_mutual_weight(edge_weights, node, other) for other in node_neighbors)
        result[node] = norm(weights) if node_neighbors else 0.0
    return result


def constraint_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute Burt's constraint keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Constraint values keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    work_store = to_undirected_store(store)
    numeric_policy = work_store.numeric_policy
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_sets_with_self(work_store)
    edge_weights = _edge_weight_map(work_store, nan_policy=resolved_nan_policy)
    scale = _neighbor_scale(edge_weights, neighbors, norm=sum)

    result: dict[Hashable, float] = {}
    for node_idx, node_neighbors in neighbors.items():
        node_id = work_store.index_to_id[node_idx]
        if not node_neighbors or all(other == node_idx for other in node_neighbors):
            result[node_id] = float("nan")
            continue
        local_sum = 0.0
        for neighbor in node_neighbors:
            direct = _normalized_mutual_weight(edge_weights, node_idx, neighbor, scale=scale)
            indirect = 0.0
            for other in node_neighbors:
                indirect += _normalized_mutual_weight(
                    edge_weights,
                    node_idx,
                    other,
                    scale=scale,
                ) * _normalized_mutual_weight(
                    edge_weights,
                    other,
                    neighbor,
                    scale=scale,
                )
            local_sum += (direct + indirect) ** 2
        result[node_id] = local_sum
    return _normalize_float_mapping(
        result,
        nan_policy=resolved_nan_policy,
        abs_tol=numeric_policy.constraint_abs_tol,
        rel_tol=numeric_policy.constraint_rel_tol,
    )


def _effective_size_unweighted(
    store: RxGraphStore,
    neighbors: Mapping[int, Sequence[int]],
    *,
    nan_policy: NanPolicy,
) -> dict[Hashable, float]:
    edge_weights = _edge_weight_map(store, nan_policy=nan_policy)
    result: dict[Hashable, float] = {}
    for node_idx, node_neighbors in neighbors.items():
        node_id = store.index_to_id[node_idx]
        filtered = [nbr for nbr in node_neighbors if nbr != node_idx]
        if not filtered:
            result[node_id] = float("nan")
            continue
        tie_count = 0
        for i, left in enumerate(filtered):
            for right in filtered[i + 1 :]:
                if _edge_weight_lookup(edge_weights, left, right) > 0:
                    tie_count += 1
        size = len(filtered)
        result[node_id] = size - (2.0 * tie_count) / size
    return result


def effective_size_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute effective size keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Effective size values keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    work_store = to_undirected_store(store)
    numeric_policy = work_store.numeric_policy
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_sets_with_self(work_store)
    if weight is None:
        result = _effective_size_unweighted(work_store, neighbors, nan_policy=resolved_nan_policy)
        return _normalize_float_mapping(
            result,
            nan_policy=resolved_nan_policy,
            abs_tol=numeric_policy.effective_abs_tol,
            rel_tol=numeric_policy.effective_rel_tol,
        )

    edge_weights = _edge_weight_map(work_store, nan_policy=resolved_nan_policy)
    sum_scale = _neighbor_scale(edge_weights, neighbors, norm=sum)
    max_scale = _neighbor_scale(edge_weights, neighbors, norm=max)

    result: dict[Hashable, float] = {}
    for node_idx, node_neighbors in neighbors.items():
        node_id = work_store.index_to_id[node_idx]
        if not node_neighbors or all(other == node_idx for other in node_neighbors):
            result[node_id] = float("nan")
            continue
        redundancy_sum = 0.0
        for neighbor in node_neighbors:
            if neighbor == node_idx:
                continue
            inner = 0.0
            for other in node_neighbors:
                inner += _normalized_mutual_weight(
                    edge_weights,
                    node_idx,
                    other,
                    scale=sum_scale,
                ) * _normalized_mutual_weight(
                    edge_weights,
                    neighbor,
                    other,
                    scale=max_scale,
                )
            redundancy_sum += 1.0 - inner
        result[node_id] = redundancy_sum
    return _normalize_float_mapping(
        result,
        nan_policy=resolved_nan_policy,
        abs_tol=numeric_policy.effective_abs_tol,
        rel_tol=numeric_policy.effective_rel_tol,
    )


def bipartite_degree_centrality_by_id(
    graph: GraphInput,
    primary: set[Hashable],
    *,
    nan_policy: NanPolicy | None = None,
) -> dict[Hashable, float]:
    """Compute bipartite degree centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Bipartite degree centrality keyed by node identifier.
    """
    store = ensure_store(graph, nan_policy=nan_policy)
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    work_store = to_undirected_store(store)
    numeric_policy = work_store.numeric_policy
    node_ids = set(work_store.node_ids())
    secondary = node_ids - primary
    primary_scale = 1.0 / float(len(secondary)) if secondary else 0.0
    secondary_scale = 1.0 / float(len(primary)) if primary else 0.0
    neighbors = _neighbor_map(work_store, include_self=False)
    result: dict[Hashable, float] = {}
    for node_id in work_store.node_ids():
        node_idx = work_store.id_to_index[node_id]
        degree = len(neighbors.get(node_idx, []))
        scale = primary_scale if node_id in primary else secondary_scale
        result[node_id] = float(degree) * scale
    return _normalize_float_mapping(
        result,
        nan_policy=resolved_nan_policy,
        abs_tol=numeric_policy.bipartite_abs_tol,
        rel_tol=numeric_policy.bipartite_rel_tol,
    )


def weighted_projection_store(
    graph: GraphInput,
    nodes: set[Hashable],
    *,
    ratio: bool = False,
    nan_policy: NanPolicy | None = None,
) -> RxGraphStore:
    """Return a weighted projection store for a subset of nodes.

    Returns
    -------
    RxGraphStore
        Weighted projection store derived from the input graph.

    Raises
    ------
    ValueError
        If projection nodes are empty or not a strict subset.
    """
    store = ensure_store(graph, nan_policy=nan_policy)
    work_store = to_undirected_store(store)
    numeric_policy = work_store.numeric_policy
    node_indices = {
        work_store.id_to_index[node_id] for node_id in nodes if node_id in work_store.id_to_index
    }
    if not node_indices:
        message = "projection nodes must be non-empty"
        raise ValueError(message)
    if len(node_indices) >= work_store.graph.num_nodes():
        message = "projection nodes must be a strict subset of graph nodes"
        raise ValueError(message)
    neighbors = _neighbor_map(work_store, include_self=False)
    projected = RxGraphStore.undirected(
        weight_policy=work_store.weight_policy,
        numeric_policy=work_store.numeric_policy,
    )
    for node_id in sorted(nodes, key=stable_key):
        if node_id in work_store.id_to_index:
            projected.ensure_node(node_id)
    other_size = work_store.graph.num_nodes() - len(node_indices)
    sorted_indices = sorted(
        node_indices,
        key=lambda idx: stable_key(work_store.index_to_id[idx]),
    )
    for left_pos, left in enumerate(sorted_indices):
        left_neighbors = set(neighbors.get(left, []))
        for right in sorted_indices[left_pos + 1 :]:
            right_neighbors = set(neighbors.get(right, []))
            common = left_neighbors & right_neighbors
            if not common:
                continue
            weight = len(common) / float(other_size) if ratio else float(len(common))
            if ratio:
                weight = _apply_tolerance(
                    weight,
                    abs_tol=numeric_policy.projection_abs_tol,
                    rel_tol=numeric_policy.projection_rel_tol,
                )
            projected.add_weighted_edge(
                work_store.index_to_id[left],
                work_store.index_to_id[right],
                weight=weight,
            )
    return projected


__all__ = [
    "BetweennessOptions",
    "GraphInput",
    "PagerankOptions",
    "RxGraph",
    "betweenness_by_id",
    "bipartite_degree_centrality_by_id",
    "closeness_by_id",
    "clustering_by_id",
    "constraint_by_id",
    "core_number_by_id",
    "degree_centrality_by_id",
    "effective_size_by_id",
    "eigenvector_centrality_by_id",
    "ensure_directed_store",
    "ensure_store",
    "graph_edge_count",
    "graph_node_count",
    "graph_to_store",
    "harmonic_centrality_by_id",
    "in_degree_centrality_by_id",
    "out_degree_centrality_by_id",
    "pagerank_by_id",
    "to_directed_store",
    "to_undirected_store",
    "triangles_by_id",
    "weighted_projection_store",
]
