"""Rustworkx-first algorithm wrappers with deterministic outputs."""

from __future__ import annotations

import contextlib
import heapq
import inspect
import math
import os
import random
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import rustworkx as rx
from rustworkx import visit

from codeintel.build.graphs.rx.components import sort_components
from codeintel.build.graphs.rx.convert import store_from_rx
from codeintel.build.graphs.rx.iterators import (
    iter_edge_payloads,
    neighbors_by_index,
    weighted_neighbors_by_index,
)
from codeintel.build.graphs.rx.normalize import (
    NanPolicy,
    normalize_mapping,
    sorted_mapping,
    sorted_nested_mapping,
    stable_key,
)
from codeintel.build.graphs.rx.policies import GraphNumericPolicy
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.graphs.rx.weights import (
    DEFAULT_WEIGHT_EPSILON,
    WeightSemantics,
    edge_cost_from_payload,
    edge_strength_from_payload,
)

RxGraph = rx.PyGraph | rx.PyDiGraph
DirectedRxGraph = rx.PyDiGraph
UndirectedRxGraph = rx.PyGraph
GraphInput = RxGraphStore | RxGraph

_MIN_BETWEENNESS_NODES = 2
_MIN_CLUSTERING_DEGREE = 2
_HITS_RESULT_LENGTH = 2
_RAYON_ENV_VAR = "RAYON_NUM_THREADS"
_RAYON_THREADS_STATE: dict[str, int | None] = {"threads": None}


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
        return sorted_mapping(normalized)
    adjusted = {
        key: _apply_tolerance(value, abs_tol=abs_tol, rel_tol=rel_tol)
        for key, value in normalized.items()
    }
    return sorted_mapping(adjusted)


def _call_with_supported_kwargs(
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return fn(*args, **filtered)


def _sorted_node_indices(store: RxGraphStore) -> list[int]:
    return [store.id_to_index[node_id] for node_id in store.node_ids()]


def _constant_weight_fn(_payload: object) -> float:
    return 1.0


def constant_weight_fn() -> Callable[[object], float]:
    """Return a constant weight function for unweighted algorithms.

    Returns
    -------
    Callable[[object], float]
        Weight function that always returns 1.0.
    """
    return _constant_weight_fn


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


@dataclass(frozen=True, slots=True)
class EdgeBetweennessOptions:
    """Options for edge betweenness centrality computation."""

    normalized: bool = True
    nan_policy: NanPolicy | None = None


@dataclass(frozen=True, slots=True)
class EigenvectorOptions:
    """Options for eigenvector centrality computation."""

    max_iter: int = 100
    tol: float = 1e-6
    weight: str | None = None
    nan_policy: NanPolicy | None = None


@dataclass(frozen=True, slots=True)
class HitsOptions:
    """Options for HITS computation."""

    max_iter: int = 100
    tol: float = 1e-6
    normalized: bool = True
    weight: str | None = None
    nan_policy: NanPolicy | None = None


@dataclass(frozen=True, slots=True)
class KatzOptions:
    """Options for Katz centrality computation."""

    alpha: float = 0.1
    beta: float = 1.0
    max_iter: int = 100
    tol: float = 1e-6
    weight: str | None = None
    nan_policy: NanPolicy | None = None


@dataclass(frozen=True, slots=True)
class GraphAlgoConfig:
    """Shared algorithm configuration for rustworkx wrappers."""

    parallel_threshold: int | None = None
    rayon_threads: int | None = None
    weight_semantics: WeightSemantics | None = None
    weight_epsilon: float = DEFAULT_WEIGHT_EPSILON


@dataclass(frozen=True, slots=True)
class WeightContext:
    """Resolved weight semantics for algorithm execution."""

    nan_policy: NanPolicy
    semantics: WeightSemantics
    epsilon: float


@dataclass
class _BfsDepthVisitor(visit.BFSVisitor):
    """BFS visitor that records bounded hop distances."""

    distances: dict[int, int]
    max_depth: int | None

    def tree_edge(self, e: tuple[int, int, object]) -> None:
        src_idx, dst_idx, _payload = e
        parent_distance = self.distances.get(src_idx, 0)
        distance = parent_distance + 1
        if self.max_depth is not None and distance > self.max_depth:
            raise visit.PruneSearch
        self.distances[dst_idx] = distance


def _apply_rayon_threads(config: GraphAlgoConfig | None) -> None:
    if config is None or config.rayon_threads is None:
        return
    if config.rayon_threads < 1:
        message = "rayon_threads must be >= 1"
        raise ValueError(message)
    if config.rayon_threads != _RAYON_THREADS_STATE["threads"]:
        os.environ[_RAYON_ENV_VAR] = str(config.rayon_threads)
        _RAYON_THREADS_STATE["threads"] = config.rayon_threads


def _resolve_parallel_threshold(config: GraphAlgoConfig | None) -> int | None:
    if config is None or config.parallel_threshold is None:
        return None
    if config.parallel_threshold < 0:
        message = "parallel_threshold must be >= 0"
        raise ValueError(message)
    return config.parallel_threshold


def _resolve_weight_semantics(
    store: RxGraphStore,
    config: GraphAlgoConfig | None,
) -> WeightSemantics:
    if config is not None and config.weight_semantics is not None:
        return config.weight_semantics
    return store.weight_policy.semantics


def _resolve_weight_epsilon(config: GraphAlgoConfig | None) -> float:
    if config is None:
        return DEFAULT_WEIGHT_EPSILON
    return config.weight_epsilon


def resolve_weight_semantics(
    store: RxGraphStore,
    config: GraphAlgoConfig | None,
) -> WeightSemantics:
    """Resolve the effective weight semantics for a store/config pair.

    Returns
    -------
    WeightSemantics
        Effective weight semantics for the store/config pair.
    """
    return _resolve_weight_semantics(store, config)


def resolve_weight_context(
    store: RxGraphStore,
    *,
    algo_config: GraphAlgoConfig | None,
    nan_policy: NanPolicy | None = None,
) -> WeightContext:
    """Resolve weight semantics, epsilon, and NaN handling for a store/config pair.

    Returns
    -------
    WeightContext
        Resolved weight semantics bundle for the store/config pair.
    """
    resolved_nan_policy = _resolve_nan_policy(store, nan_policy)
    semantics = _resolve_weight_semantics(store, algo_config)
    epsilon = _resolve_weight_epsilon(algo_config)
    return WeightContext(
        nan_policy=resolved_nan_policy,
        semantics=semantics,
        epsilon=epsilon,
    )


def resolve_weight_epsilon(config: GraphAlgoConfig | None) -> float:
    """Resolve the effective epsilon used for weight conversions.

    Returns
    -------
    float
        Effective epsilon for weight conversions.
    """
    return _resolve_weight_epsilon(config)


def edge_strength_weight_fn(*, context: WeightContext) -> Callable[[object], float]:
    """Return a weight function that yields edge strengths.

    Returns
    -------
    Callable[[object], float]
        Weight function yielding edge strengths.
    """
    return _edge_strength_fn(
        nan_policy=context.nan_policy,
        semantics=context.semantics,
        epsilon=context.epsilon,
    )


def edge_cost_weight_fn(*, context: WeightContext) -> Callable[[object], float]:
    """Return a weight function that yields edge costs.

    Returns
    -------
    Callable[[object], float]
        Weight function yielding edge costs.
    """
    return _edge_cost_fn(
        nan_policy=context.nan_policy,
        semantics=context.semantics,
        epsilon=context.epsilon,
    )


def edge_strength_from_context(
    payload: object | None,
    *,
    context: WeightContext,
) -> float:
    """Return an edge strength weight from a payload using a resolved context.

    Returns
    -------
    float
        Edge strength weight for algorithm inputs.
    """
    return edge_strength_from_payload(
        payload,
        nan_policy=context.nan_policy,
        semantics=context.semantics,
        epsilon=context.epsilon,
    )


def edge_cost_from_context(
    payload: object | None,
    *,
    context: WeightContext,
) -> float:
    """Return an edge cost weight from a payload using a resolved context.

    Returns
    -------
    float
        Edge cost weight for algorithm inputs.
    """
    return edge_cost_from_payload(
        payload,
        nan_policy=context.nan_policy,
        semantics=context.semantics,
        epsilon=context.epsilon,
    )


def _edge_strength_fn(
    *,
    nan_policy: NanPolicy,
    semantics: WeightSemantics,
    epsilon: float,
) -> Callable[[object], float]:
    def weight_fn(payload: object) -> float:
        return edge_strength_from_payload(
            payload,
            nan_policy=nan_policy,
            semantics=semantics,
            epsilon=epsilon,
        )

    return weight_fn


def _edge_cost_fn(
    *,
    nan_policy: NanPolicy,
    semantics: WeightSemantics,
    epsilon: float,
) -> Callable[[object], float]:
    def cost_fn(payload: object) -> float:
        return edge_cost_from_payload(
            payload,
            nan_policy=nan_policy,
            semantics=semantics,
            epsilon=epsilon,
        )

    return cost_fn


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


def in_degree_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Return in-degree counts keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Mapping of node ids to in-degree counts.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    directed_graph = _directed_graph(store)
    mapping = {
        node_id: int(directed_graph.in_degree(store.id_to_index[node_id]))
        for node_id in store.node_ids()
    }
    return sorted_mapping(mapping)


def out_degree_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Return out-degree counts keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Mapping of node ids to out-degree counts.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    directed_graph = _directed_graph(store)
    mapping = {
        node_id: int(directed_graph.out_degree(store.id_to_index[node_id]))
        for node_id in store.node_ids()
    }
    return sorted_mapping(mapping)


def degree_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Return undirected degree counts keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Mapping of node ids to undirected degree counts.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    work_store = to_undirected_store(store)
    undirected_graph = _undirected_graph(work_store)
    mapping = {
        node_id: int(undirected_graph.degree(work_store.id_to_index[node_id]))
        for node_id in work_store.node_ids()
    }
    return sorted_mapping(mapping)


def total_degree_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Return total degree counts keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Mapping of node ids to total degree counts.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    if store.is_directed:
        directed_graph = _directed_graph(store)
        mapping = {
            node_id: int(
                directed_graph.in_degree(store.id_to_index[node_id])
                + directed_graph.out_degree(store.id_to_index[node_id])
            )
            for node_id in store.node_ids()
        }
        return sorted_mapping(mapping)
    undirected_graph = _undirected_graph(store)
    mapping = {
        node_id: int(undirected_graph.degree(store.id_to_index[node_id]))
        for node_id in store.node_ids()
    }
    return sorted_mapping(mapping)


def successors_by_id(
    graph: GraphInput,
    node_id: Hashable,
) -> list[Hashable]:
    """Return successor node ids with stable ordering.

    Returns
    -------
    list[Hashable]
        Successor node ids ordered by stable key.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    node_idx = store.id_to_index.get(node_id)
    if node_idx is None:
        return []
    directed_graph = _directed_graph(store)
    successors = [store.index_to_id[idx] for idx in directed_graph.successor_indices(node_idx)]
    return sorted(successors, key=stable_key)


def predecessors_by_id(
    graph: GraphInput,
    node_id: Hashable,
) -> list[Hashable]:
    """Return predecessor node ids with stable ordering.

    Returns
    -------
    list[Hashable]
        Predecessor node ids ordered by stable key.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    node_idx = store.id_to_index.get(node_id)
    if node_idx is None:
        return []
    directed_graph = _directed_graph(store)
    predecessors = [store.index_to_id[idx] for idx in directed_graph.predecessor_indices(node_idx)]
    return sorted(predecessors, key=stable_key)


def insert_node_on_out_edges_by_id(
    store: RxGraphStore,
    node_id: Hashable,
    ref_node_id: Hashable,
    *,
    attrs: Mapping[str, object] | None = None,
) -> int | None:
    """Insert an existing node between a reference node and its successors.

    Returns
    -------
    int | None
        Node index for the inserted node, or None if the reference node is missing.

    Raises
    ------
    ValueError
        If the store is not directed.
    """
    if not store.is_directed:
        message = "insert_node_on_out_edges_by_id requires a directed graph store"
        raise ValueError(message)
    ref_idx = store.id_to_index.get(ref_node_id)
    if ref_idx is None:
        return None
    node_idx = store.ensure_node(node_id)
    if attrs:
        store.set_node_attrs(node_id, attrs)
    directed_graph = _directed_graph(store)
    directed_graph.insert_node_on_out_edges(node_idx, ref_idx)
    store.touch()
    return node_idx


def remove_node_retain_edges_by_id(
    store: RxGraphStore,
    node_id: Hashable,
    *,
    use_outgoing: bool = False,
    condition: Callable[[object, object], bool] | None = None,
) -> bool:
    """Remove a node while retaining predecessor-to-successor edges.

    Returns
    -------
    bool
        True when the node was removed.

    Raises
    ------
    ValueError
        If the store is not directed.
    """
    if not store.is_directed:
        message = "remove_node_retain_edges_by_id requires a directed graph store"
        raise ValueError(message)
    node_idx = store.id_to_index.get(node_id)
    if node_idx is None:
        return False
    directed_graph = _directed_graph(store)
    directed_graph.remove_node_retain_edges(
        node_idx,
        use_outgoing=use_outgoing,
        condition=condition,
    )
    store.id_to_index.pop(node_id, None)
    store.index_to_id.pop(node_idx, None)
    store.node_attrs.pop(node_id, None)
    store.touch()
    return True


def pagerank_by_id(
    graph: GraphInput,
    *,
    options: PagerankOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute PageRank scores keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        PageRank scores keyed by node identifier.
    """
    resolved = options or PagerankOptions()
    store = ensure_directed_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=resolved.nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    directed_graph = _directed_graph(store)
    weight_fn: Callable[[object], float] | None = None
    if resolved.weight is not None:
        weight_fn = edge_strength_weight_fn(context=weight_ctx)
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
    options: EigenvectorOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute eigenvector centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Eigenvector centrality scores keyed by node identifier.
    """
    resolved = options or EigenvectorOptions()
    store = ensure_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=resolved.nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return {}
    undirected_graph = _undirected_graph(work_store)
    weight_fn: Callable[[object], float] | None = None
    if resolved.weight is not None:
        weight_fn = edge_strength_weight_fn(context=weight_ctx)
    raw = rx.graph_eigenvector_centrality(
        undirected_graph,
        weight_fn=weight_fn,
        max_iter=resolved.max_iter,
        tol=resolved.tol,
    )
    mapped = {work_store.index_to_id[idx]: float(score) for idx, score in raw.items()}
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def _map_index_scores(
    store: RxGraphStore,
    scores: Mapping[object, float],
) -> dict[Hashable, float]:
    mapped: dict[Hashable, float] = {}
    for idx, score in scores.items():
        node_id = store.index_to_id.get(idx, idx) if isinstance(idx, int) else idx
        mapped[node_id] = float(score)
    return mapped


def _require_hits_result(
    raw: object,
) -> tuple[Mapping[object, float], Mapping[object, float]]:
    if isinstance(raw, tuple) and len(raw) == _HITS_RESULT_LENGTH:
        hubs, authorities = raw
        if isinstance(hubs, Mapping) and isinstance(authorities, Mapping):
            return hubs, authorities
    msg = "rustworkx HITS returned an unexpected result."
    raise TypeError(msg)


def hits_by_id(
    graph: GraphInput,
    *,
    options: HitsOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> tuple[dict[Hashable, float], dict[Hashable, float]]:
    """Compute HITS hub/authority scores keyed by node id.

    Returns
    -------
    tuple[dict[Hashable, float], dict[Hashable, float]]
        Hub scores and authority scores keyed by node identifier.
    """
    resolved = options or HitsOptions()
    store = ensure_directed_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=resolved.nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    if store.graph.num_nodes() == 0:
        return {}, {}
    hits_fn = getattr(rx, "hits", None)
    if not callable(hits_fn):
        msg = "rustworkx.hits is not available in this environment."
        raise NotImplementedError(msg)
    weight_fn: Callable[[object], float] | None = None
    if resolved.weight is not None:
        weight_fn = edge_strength_weight_fn(context=weight_ctx)
    raw = _call_with_supported_kwargs(
        hits_fn,
        _directed_graph(store),
        max_iter=resolved.max_iter,
        tol=resolved.tol,
        normalized=resolved.normalized,
        weight_fn=weight_fn,
    )
    hubs_raw, authorities_raw = _require_hits_result(raw)
    hubs = _normalize_float_mapping(
        _map_index_scores(store, hubs_raw),
        nan_policy=resolved_nan_policy,
    )
    authorities = _normalize_float_mapping(
        _map_index_scores(store, authorities_raw),
        nan_policy=resolved_nan_policy,
    )
    return hubs, authorities


def _katz_fn_for_store(store: RxGraphStore) -> Callable[..., object]:
    candidates = (
        ("digraph_katz_centrality", "katz_centrality")
        if store.is_directed
        else ("graph_katz_centrality", "katz_centrality")
    )
    for name in candidates:
        candidate = getattr(rx, name, None)
        if callable(candidate):
            return candidate
    msg = "rustworkx Katz centrality is not available in this environment."
    raise NotImplementedError(msg)


def katz_centrality_by_id(
    graph: GraphInput,
    *,
    options: KatzOptions | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute Katz centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Katz centrality scores keyed by node identifier.

    Raises
    ------
    TypeError
        Raised when rustworkx returns an unexpected result type.
    """
    resolved = options or KatzOptions()
    store = ensure_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=resolved.nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    if store.graph.num_nodes() == 0:
        return {}
    weight_fn: Callable[[object], float] | None = None
    if resolved.weight is not None:
        weight_fn = edge_strength_weight_fn(context=weight_ctx)
    katz_fn = _katz_fn_for_store(store)
    graph_obj = _directed_graph(store) if store.is_directed else _undirected_graph(store)
    raw = _call_with_supported_kwargs(
        katz_fn,
        graph_obj,
        alpha=resolved.alpha,
        beta=resolved.beta,
        max_iter=resolved.max_iter,
        tol=resolved.tol,
        weight_fn=weight_fn,
    )
    if not isinstance(raw, Mapping):
        msg = "rustworkx Katz centrality returned an unexpected result."
        raise TypeError(msg)
    mapped = _map_index_scores(store, raw)
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def _closeness_unweighted(
    store: RxGraphStore,
    *,
    wf_improved: bool,
    parallel_threshold: int | None,
) -> Mapping[int, float]:
    if store.is_directed:
        graph = _directed_graph(store)
        if parallel_threshold is None:
            return rx.digraph_closeness_centrality(graph, wf_improved=wf_improved)
        return rx.digraph_closeness_centrality(
            graph,
            wf_improved=wf_improved,
            parallel_threshold=parallel_threshold,
        )
    graph = _undirected_graph(store)
    if parallel_threshold is None:
        return rx.graph_closeness_centrality(graph, wf_improved=wf_improved)
    return rx.graph_closeness_centrality(
        graph,
        wf_improved=wf_improved,
        parallel_threshold=parallel_threshold,
    )


def _closeness_weighted(
    store: RxGraphStore,
    *,
    weight_fn: Callable[[object], float],
    wf_improved: bool,
    parallel_threshold: int | None,
) -> Mapping[int, float]:
    if store.is_directed:
        graph = _directed_graph(store)
        if parallel_threshold is None:
            return rx.digraph_newman_weighted_closeness_centrality(
                graph,
                weight_fn=weight_fn,
                default_weight=1.0,
                wf_improved=wf_improved,
            )
        return rx.digraph_newman_weighted_closeness_centrality(
            graph,
            weight_fn=weight_fn,
            default_weight=1.0,
            wf_improved=wf_improved,
            parallel_threshold=parallel_threshold,
        )
    graph = _undirected_graph(store)
    if parallel_threshold is None:
        return rx.graph_newman_weighted_closeness_centrality(
            graph,
            weight_fn=weight_fn,
            default_weight=1.0,
            wf_improved=wf_improved,
        )
    return rx.graph_newman_weighted_closeness_centrality(
        graph,
        weight_fn=weight_fn,
        default_weight=1.0,
        wf_improved=wf_improved,
        parallel_threshold=parallel_threshold,
    )


def closeness_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    wf_improved: bool = True,
    nan_policy: NanPolicy | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute closeness centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Closeness centrality scores keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    parallel_threshold = _resolve_parallel_threshold(algo_config)
    if store.graph.num_nodes() == 0:
        return {}
    if weight is None:
        raw = _closeness_unweighted(
            store,
            wf_improved=wf_improved,
            parallel_threshold=parallel_threshold,
        )
    else:
        weight_fn = edge_strength_weight_fn(context=weight_ctx)
        raw = _closeness_weighted(
            store,
            weight_fn=weight_fn,
            wf_improved=wf_improved,
            parallel_threshold=parallel_threshold,
        )
    mapped = {store.index_to_id[idx]: float(score) for idx, score in raw.items()}
    return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)


def _edge_weight_map(
    store: RxGraphStore,
    *,
    context: WeightContext,
) -> dict[tuple[int, int], float]:
    edge_map: dict[tuple[int, int], float] = {}
    for src_idx, dst_idx, payload in iter_edge_payloads(store):
        if store.is_directed:
            key = (src_idx, dst_idx)
        else:
            key = (min(src_idx, dst_idx), max(src_idx, dst_idx))
        edge_map[key] = edge_strength_from_context(payload, context=context)
    return edge_map


def _edge_cost_map(
    store: RxGraphStore,
    *,
    context: WeightContext,
) -> dict[tuple[int, int], float]:
    edge_map: dict[tuple[int, int], float] = {}
    for src_idx, dst_idx, payload in iter_edge_payloads(store):
        if store.is_directed:
            key = (src_idx, dst_idx)
        else:
            key = (min(src_idx, dst_idx), max(src_idx, dst_idx))
        edge_map[key] = edge_cost_from_context(payload, context=context)
    return edge_map


def _neighbor_map(
    store: RxGraphStore,
    *,
    include_self: bool = False,
) -> dict[int, list[int]]:
    return neighbors_by_index(store, include_self=include_self)


def _weighted_neighbor_map(
    store: RxGraphStore,
    edge_weights: Mapping[tuple[int, int], float],
) -> dict[int, list[tuple[int, float]]]:
    return weighted_neighbors_by_index(store, edge_weights)


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
    parallel_threshold: int | None,
) -> dict[Hashable, float]:
    if store.is_directed:
        directed_graph = _directed_graph(store)
        if parallel_threshold is None:
            raw = rx.digraph_betweenness_centrality(
                directed_graph,
                normalized=normalized,
            )
        else:
            raw = rx.digraph_betweenness_centrality(
                directed_graph,
                normalized=normalized,
                parallel_threshold=parallel_threshold,
            )
    else:
        undirected_graph = _undirected_graph(store)
        if parallel_threshold is None:
            raw = rx.graph_betweenness_centrality(
                undirected_graph,
                normalized=normalized,
            )
        else:
            raw = rx.graph_betweenness_centrality(
                undirected_graph,
                normalized=normalized,
                parallel_threshold=parallel_threshold,
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
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute betweenness centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Betweenness centrality scores keyed by node identifier.
    """
    resolved = options or BetweennessOptions()
    store = ensure_store(graph, weight=resolved.weight, nan_policy=resolved.nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=resolved.nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    parallel_threshold = _resolve_parallel_threshold(algo_config)
    numeric_policy = store.numeric_policy
    node_count = store.graph.num_nodes()
    if node_count == 0:
        return {}
    if resolved.weight is None and resolved.k is None:
        mapped = _betweenness_builtin_by_id(
            store,
            normalized=resolved.normalized,
            parallel_threshold=parallel_threshold,
        )
        return _normalize_float_mapping(mapped, nan_policy=resolved_nan_policy)

    indices, sampled = _resolve_sampled_indices(store, k=resolved.k, seed=resolved.seed)

    if resolved.weight is None:
        neighbors = _neighbor_map(store, include_self=False)
        betweenness = _brandes_unweighted(neighbors, indices)
    else:
        edge_weights = _edge_cost_map(store, context=weight_ctx)
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
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute harmonic centrality keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Harmonic centrality scores keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    _apply_rayon_threads(algo_config)
    numeric_policy = store.numeric_policy
    if store.graph.num_nodes() == 0:
        return {}
    weight_fn: Callable[[object], float] = _constant_weight_fn
    if weight is not None:
        weight_fn = edge_cost_weight_fn(context=weight_ctx)
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
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute clustering coefficients keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Clustering coefficients keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    work_store = to_undirected_store(store)
    weight_ctx = resolve_weight_context(
        work_store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    numeric_policy = work_store.numeric_policy
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_map(work_store, include_self=False)
    if weight is None:
        result = _clustering_unweighted(work_store, neighbors)
        return _normalize_float_mapping(result, nan_policy=resolved_nan_policy)
    result = _clustering_weighted(
        work_store,
        neighbors,
        context=weight_ctx,
    )
    return _normalize_float_mapping(
        result,
        nan_policy=resolved_nan_policy,
        abs_tol=numeric_policy.clustering_abs_tol,
        rel_tol=numeric_policy.clustering_rel_tol,
    )


def transitivity_score(graph: GraphInput) -> float:
    """Compute global transitivity for an undirected view of the graph.

    Returns
    -------
    float
        Global transitivity (global clustering coefficient).
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return 0.0
    neighbors = _neighbor_map(work_store, include_self=False)
    triangle_total = sum(_triangle_counts(neighbor_map=neighbors).values())
    if triangle_total == 0:
        return 0.0
    triplets = 0.0
    for neighbor_list in neighbors.values():
        degree = len(neighbor_list)
        if degree < _MIN_CLUSTERING_DEGREE:
            continue
        triplets += degree * (degree - 1) / 2
    if triplets == 0.0:
        return 0.0
    return float(triangle_total) / triplets


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
    context: WeightContext,
) -> dict[Hashable, float]:
    edge_weights = _edge_weight_map(store, context=context)
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
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute Burt's constraint keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Constraint values keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    work_store = to_undirected_store(store)
    weight_ctx = resolve_weight_context(
        work_store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    numeric_policy = work_store.numeric_policy
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_sets_with_self(work_store)
    edge_weights = _edge_weight_map(
        work_store,
        context=weight_ctx,
    )
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
    context: WeightContext,
) -> dict[Hashable, float]:
    edge_weights = _edge_weight_map(store, context=context)
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
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Compute effective size keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Effective size values keyed by node identifier.
    """
    store = ensure_store(graph, weight=weight, nan_policy=nan_policy)
    work_store = to_undirected_store(store)
    weight_ctx = resolve_weight_context(
        work_store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    resolved_nan_policy = weight_ctx.nan_policy
    numeric_policy = work_store.numeric_policy
    if work_store.graph.num_nodes() == 0:
        return {}
    neighbors = _neighbor_sets_with_self(work_store)
    if weight is None:
        result = _effective_size_unweighted(
            work_store,
            neighbors,
            context=weight_ctx,
        )
        return _normalize_float_mapping(
            result,
            nan_policy=resolved_nan_policy,
            abs_tol=numeric_policy.effective_abs_tol,
            rel_tol=numeric_policy.effective_rel_tol,
        )

    edge_weights = _edge_weight_map(work_store, context=weight_ctx)
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


def connected_components_by_id(graph: GraphInput) -> list[set[Hashable]]:
    """Return connected components keyed by node id (undirected).

    Returns
    -------
    list[set[Hashable]]
        Components as sets of node identifiers.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []
    undirected_graph = _undirected_graph(work_store)
    components = [set(comp) for comp in rx.connected_components(undirected_graph)]
    sorted_components = sort_components(work_store, components)
    return [{work_store.index_to_id[idx] for idx in component} for component in sorted_components]


def weakly_connected_components_by_id(graph: GraphInput) -> list[set[Hashable]]:
    """Return weakly connected components keyed by node id.

    Returns
    -------
    list[set[Hashable]]
        Components as sets of node identifiers.
    """
    store = ensure_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    if not store.is_directed:
        return connected_components_by_id(store)
    directed_graph = _directed_graph(store)
    components = [set(comp) for comp in rx.weakly_connected_components(directed_graph)]
    sorted_components = sort_components(store, components)
    return [{store.index_to_id[idx] for idx in component} for component in sorted_components]


def strongly_connected_components_by_id(graph: GraphInput) -> list[set[Hashable]]:
    """Return strongly connected components keyed by node id.

    Returns
    -------
    list[set[Hashable]]
        Components as sets of node identifiers.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    directed_graph = _directed_graph(store)
    components = [set(comp) for comp in rx.strongly_connected_components(directed_graph)]
    sorted_components = sort_components(store, components)
    return [{store.index_to_id[idx] for idx in component} for component in sorted_components]


def bridges_by_id(graph: GraphInput) -> list[tuple[Hashable, Hashable]]:
    """Return bridge edges keyed by node id.

    Returns
    -------
    list[tuple[Hashable, Hashable]]
        Bridge edge endpoints.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []
    undirected_graph = _undirected_graph(work_store)
    bridges: list[tuple[Hashable, Hashable]] = []
    for edge in rx.bridges(undirected_graph):
        src_idx, dst_idx = cast("tuple[int, int]", edge)
        bridges.append((work_store.index_to_id[src_idx], work_store.index_to_id[dst_idx]))
    return sorted(bridges, key=stable_key)


def articulation_points_by_id(graph: GraphInput) -> list[Hashable]:
    """Return articulation points keyed by node id.

    Returns
    -------
    list[Hashable]
        Articulation point node identifiers.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []
    undirected_graph = _undirected_graph(work_store)
    points = [work_store.index_to_id[idx] for idx in rx.articulation_points(undirected_graph)]
    return sorted(points, key=stable_key)


def simple_cycles_by_id(
    graph: GraphInput,
    *,
    limit: int | None = None,
) -> list[list[Hashable]]:
    """Return simple cycles keyed by node id.

    Returns
    -------
    list[list[Hashable]]
        Cycles as ordered node id lists.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    directed_graph = _directed_graph(store)
    cycles: list[list[Hashable]] = []
    for cycle in rx.simple_cycles(directed_graph):
        cycles.append([store.index_to_id[idx] for idx in cycle])
        if limit is not None and len(cycles) >= limit:
            break
    cycles.sort(key=lambda path: stable_key(tuple(path)))
    return cycles


def topological_generations_by_id(graph: GraphInput) -> list[list[Hashable]]:
    """Return topological generations keyed by node id.

    Returns
    -------
    list[list[Hashable]]
        Generation lists ordered by stable node identifiers.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    directed_graph = _directed_graph(store)
    generations: list[list[Hashable]] = []
    for generation in rx.topological_generations(directed_graph):
        ordered = sorted(
            generation,
            key=lambda idx: stable_key(store.index_to_id[idx]),
        )
        generations.append([store.index_to_id[idx] for idx in ordered])
    return generations


def topological_layers_by_id(graph: GraphInput) -> dict[Hashable, int]:
    """Return topological layers keyed by node id.

    Returns
    -------
    dict[Hashable, int]
        Node id to layer mapping.
    """
    layers: dict[Hashable, int] = {}
    for layer, generation in enumerate(topological_generations_by_id(graph)):
        for node_id in generation:
            layers[node_id] = layer
    return sorted_mapping(layers)


def is_directed_acyclic(graph: GraphInput) -> bool:
    """Return True when the directed graph is acyclic.

    Returns
    -------
    bool
        True when the directed graph has no cycles.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return True
    directed_graph = _directed_graph(store)
    try:
        return rx.is_directed_acyclic_graph(directed_graph)
    except rx.NullGraph:
        return False


def dag_longest_path_length(
    graph: GraphInput,
    *,
    allow_condensation: bool = True,
) -> int:
    """Return longest path length for a DAG (condensing if needed).

    Returns
    -------
    int
        Longest path length, or 0 when unavailable.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return 0
    directed_graph = _directed_graph(store)
    try:
        if rx.is_directed_acyclic_graph(directed_graph):
            return int(rx.dag_longest_path_length(directed_graph))
    except rx.NullGraph:
        return 0
    if not allow_condensation:
        return 0
    condensed = cast("rx.PyDiGraph", rx.condensation(directed_graph))
    try:
        return int(rx.dag_longest_path_length(condensed))
    except (rx.DAGHasCycle, rx.NullGraph):
        return 0


def graph_distance_matrix(graph: GraphInput) -> list[list[float]]:
    """Return an undirected graph distance matrix.

    Returns
    -------
    list[list[float]]
        Distance matrix for the undirected graph.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return []
    undirected_graph = _undirected_graph(work_store)
    try:
        return list(rx.graph_distance_matrix(undirected_graph))
    except rx.NullGraph:
        return []


def graph_unweighted_average_shortest_path_length(
    graph: GraphInput,
) -> float | None:
    """Return average shortest path length for an undirected graph.

    Returns
    -------
    float | None
        Average shortest path length, or None when unavailable.
    """
    store = ensure_store(graph)
    work_store = to_undirected_store(store)
    if work_store.graph.num_nodes() == 0:
        return None
    undirected_graph = _undirected_graph(work_store)
    try:
        return float(rx.graph_unweighted_average_shortest_path_length(undirected_graph))
    except rx.NullGraph:
        return None


def ancestors_by_id(
    graph: GraphInput,
    source: Hashable,
    *,
    include_source: bool = False,
) -> set[Hashable]:
    """Return ancestor nodes for a source id.

    Returns
    -------
    set[Hashable]
        Ancestor node ids (optionally including the source).
    """
    store = ensure_directed_store(graph)
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return {source} if include_source else set()
    directed_graph = _directed_graph(store)
    try:
        ancestors = rx.ancestors(directed_graph, source_idx)
    except (rx.InvalidNode, rx.NullGraph):
        ancestors = set()
    result = {store.index_to_id[idx] for idx in ancestors}
    if include_source:
        result.add(source)
    return result


def descendants_by_id(
    graph: GraphInput,
    source: Hashable,
    *,
    include_source: bool = False,
) -> set[Hashable]:
    """Return descendant nodes for a source id.

    Returns
    -------
    set[Hashable]
        Descendant node ids (optionally including the source).
    """
    store = ensure_directed_store(graph)
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return {source} if include_source else set()
    directed_graph = _directed_graph(store)
    try:
        descendants = rx.descendants(directed_graph, source_idx)
    except (rx.InvalidNode, rx.NullGraph):
        descendants = set()
    result = {store.index_to_id[idx] for idx in descendants}
    if include_source:
        result.add(source)
    return result


def bfs_distances_by_id(
    graph: GraphInput,
    source: Hashable,
    *,
    max_depth: int | None = None,
) -> dict[Hashable, int]:
    """Return BFS hop distances from a source node id.

    Returns
    -------
    dict[Hashable, int]
        Mapping of node ids to hop distances.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return {}
    distances: dict[int, int] = {source_idx: 0}
    visitor = _BfsDepthVisitor(distances=distances, max_depth=max_depth)
    with contextlib.suppress(visit.PruneSearch):
        rx.bfs_search(_directed_graph(store), [source_idx], visitor)
    return {store.index_to_id[idx]: dist for idx, dist in distances.items()}


def digraph_shortest_path_lengths_by_id(
    graph: GraphInput,
    source: Hashable,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, float]:
    """Return shortest path lengths from a source keyed by node id.

    Returns
    -------
    dict[Hashable, float]
        Mapping of node ids to shortest path lengths.
    """
    store = ensure_directed_store(graph, weight=weight, nan_policy=nan_policy)
    if store.graph.num_nodes() == 0:
        return {}
    source_idx = store.id_to_index.get(source)
    if source_idx is None:
        return {}
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    weight_fn = constant_weight_fn()
    if weight is not None:
        weight_fn = edge_cost_weight_fn(context=weight_ctx)
    directed_graph = _directed_graph(store)
    try:
        lengths = rx.digraph_dijkstra_shortest_path_lengths(
            directed_graph,
            source_idx,
            weight_fn,
        )
    except (rx.InvalidNode, rx.NullGraph):
        return {}
    mapped = {store.index_to_id[idx]: float(dist) for idx, dist in lengths.items()}
    return _normalize_float_mapping(mapped, nan_policy=weight_ctx.nan_policy)


def digraph_all_pairs_shortest_path_lengths_by_id(
    graph: GraphInput,
    *,
    weight: str | None = None,
    nan_policy: NanPolicy | None = None,
    algo_config: GraphAlgoConfig | None = None,
) -> dict[Hashable, dict[Hashable, float]]:
    """Return all-pairs shortest path lengths keyed by node id.

    Returns
    -------
    dict[Hashable, dict[Hashable, float]]
        Nested mapping of source node ids to target distances.
    """
    store = ensure_directed_store(graph, weight=weight, nan_policy=nan_policy)
    if store.graph.num_nodes() == 0:
        return {}
    weight_ctx = resolve_weight_context(
        store,
        algo_config=algo_config,
        nan_policy=nan_policy,
    )
    weight_fn = constant_weight_fn()
    if weight is not None:
        weight_fn = edge_cost_weight_fn(context=weight_ctx)
    directed_graph = _directed_graph(store)
    try:
        lengths = rx.digraph_all_pairs_dijkstra_path_lengths(
            directed_graph,
            weight_fn,
        )
    except rx.NullGraph:
        return {}
    mapped: dict[Hashable, dict[Hashable, float]] = {}
    for src_idx, targets in lengths.items():
        src_id = store.index_to_id[src_idx]
        mapped[src_id] = {store.index_to_id[idx]: float(dist) for idx, dist in targets.items()}
    return sorted_nested_mapping(mapped)


def immediate_dominators_by_id(
    graph: GraphInput,
    entry: Hashable,
) -> dict[Hashable, Hashable | None]:
    """Return immediate dominators keyed by node id.

    Returns
    -------
    dict[Hashable, Hashable | None]
        Mapping of node ids to immediate dominators.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    entry_idx = store.id_to_index.get(entry)
    if entry_idx is None:
        return {}
    directed_graph = _directed_graph(store)
    try:
        idoms = rx.immediate_dominators(directed_graph, entry_idx)
    except (rx.InvalidNode, rx.NullGraph):
        return {}
    result: dict[Hashable, Hashable | None] = {}
    for node_idx, idom_idx in idoms.items():
        node_id = store.index_to_id[node_idx]
        if node_idx == entry_idx:
            result[node_id] = None
        else:
            result[node_id] = store.index_to_id[idom_idx]
    return {node: result[node] for node in sorted(result, key=stable_key)}


def dominance_frontiers_by_id(
    graph: GraphInput,
    entry: Hashable,
) -> dict[Hashable, frozenset[Hashable]]:
    """Return dominance frontiers keyed by node id.

    Returns
    -------
    dict[Hashable, frozenset[Hashable]]
        Mapping of node ids to dominance frontiers.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return {}
    entry_idx = store.id_to_index.get(entry)
    if entry_idx is None:
        return {}
    directed_graph = _directed_graph(store)
    try:
        frontiers = rx.dominance_frontiers(directed_graph, entry_idx)
    except (rx.InvalidNode, rx.NullGraph):
        return {}
    mapped = {
        store.index_to_id[node_idx]: frozenset(store.index_to_id[idx] for idx in frontier)
        for node_idx, frontier in frontiers.items()
    }
    return {node: mapped[node] for node in sorted(mapped, key=stable_key)}


def simple_paths_by_id(
    graph: GraphInput,
    source: Hashable,
    target: Hashable,
    *,
    cutoff: int,
    limit: int | None = None,
) -> list[list[Hashable]]:
    """Return simple paths between nodes keyed by node id.

    Returns
    -------
    list[list[Hashable]]
        Simple paths from source to target, ordered by discovery.
    """
    store = ensure_directed_store(graph)
    if store.graph.num_nodes() == 0:
        return []
    source_idx = store.id_to_index.get(source)
    target_idx = store.id_to_index.get(target)
    if source_idx is None or target_idx is None:
        return []
    directed_graph = _directed_graph(store)
    paths: list[list[Hashable]] = []
    try:
        for path in rx.digraph_all_simple_paths(
            directed_graph,
            source_idx,
            target_idx,
            cutoff=cutoff,
        ):
            paths.append([store.index_to_id[idx] for idx in path])
            if limit is not None and len(paths) >= limit:
                break
    except (rx.InvalidNode, rx.NoPathFound, rx.NullGraph):
        return []
    return paths


__all__ = [
    "BetweennessOptions",
    "EigenvectorOptions",
    "GraphAlgoConfig",
    "GraphInput",
    "HitsOptions",
    "KatzOptions",
    "PagerankOptions",
    "RxGraph",
    "WeightContext",
    "ancestors_by_id",
    "articulation_points_by_id",
    "betweenness_by_id",
    "bfs_distances_by_id",
    "bipartite_degree_centrality_by_id",
    "bridges_by_id",
    "closeness_by_id",
    "clustering_by_id",
    "connected_components_by_id",
    "constant_weight_fn",
    "constraint_by_id",
    "core_number_by_id",
    "dag_longest_path_length",
    "degree_by_id",
    "degree_centrality_by_id",
    "descendants_by_id",
    "digraph_all_pairs_shortest_path_lengths_by_id",
    "digraph_shortest_path_lengths_by_id",
    "dominance_frontiers_by_id",
    "edge_cost_weight_fn",
    "edge_strength_weight_fn",
    "effective_size_by_id",
    "eigenvector_centrality_by_id",
    "ensure_directed_store",
    "ensure_store",
    "graph_distance_matrix",
    "graph_edge_count",
    "graph_node_count",
    "graph_to_store",
    "graph_unweighted_average_shortest_path_length",
    "harmonic_centrality_by_id",
    "hits_by_id",
    "immediate_dominators_by_id",
    "in_degree_by_id",
    "in_degree_centrality_by_id",
    "insert_node_on_out_edges_by_id",
    "is_directed_acyclic",
    "katz_centrality_by_id",
    "out_degree_by_id",
    "out_degree_centrality_by_id",
    "pagerank_by_id",
    "predecessors_by_id",
    "remove_node_retain_edges_by_id",
    "resolve_weight_context",
    "resolve_weight_epsilon",
    "resolve_weight_semantics",
    "simple_cycles_by_id",
    "simple_paths_by_id",
    "strongly_connected_components_by_id",
    "successors_by_id",
    "to_directed_store",
    "to_undirected_store",
    "topological_generations_by_id",
    "topological_layers_by_id",
    "total_degree_by_id",
    "transitivity_score",
    "triangles_by_id",
    "weakly_connected_components_by_id",
    "weighted_projection_store",
]
