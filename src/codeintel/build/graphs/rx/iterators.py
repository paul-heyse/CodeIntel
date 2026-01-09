"""Shared iteration helpers for rustworkx graph stores."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

from codeintel.build.graphs.rx.normalize import NanPolicy, edge_weight_from_payload, stable_key
from codeintel.build.graphs.rx.store import RxGraphStore


def iter_edge_payloads(store: RxGraphStore) -> Iterable[tuple[int, int, object]]:
    """Yield edge endpoints and payloads from a rustworkx store.

    Yields
    ------
    tuple[int, int, object]
        Edge endpoints and payloads.
    """
    for (src_idx, dst_idx), payload in zip(
        store.graph.edge_list(),
        store.graph.edges(),
        strict=True,
    ):
        yield src_idx, dst_idx, payload


def iter_edge_index_payloads(store: RxGraphStore) -> Iterable[tuple[int, int, int, object]]:
    """Yield edge index, endpoints, and payloads from a rustworkx store.

    Yields
    ------
    tuple[int, int, int, object]
        Edge index, endpoints, and payloads.
    """
    for edge_idx, (src_idx, dst_idx, payload) in enumerate(iter_edge_payloads(store)):
        yield edge_idx, src_idx, dst_idx, payload


def edge_index_map(store: RxGraphStore) -> dict[tuple[int, int], int]:
    """Return a mapping of edge endpoints to edge indices.

    Returns
    -------
    dict[tuple[int, int], int]
        Mapping of edge endpoints to edge index in edge_list ordering.
    """
    indices: dict[tuple[int, int], int] = {}
    for edge_idx, src_idx, dst_idx, _payload in iter_edge_index_payloads(store):
        indices[src_idx, dst_idx] = edge_idx
    return indices


def iter_edge_id_payloads(store: RxGraphStore) -> Iterable[tuple[Hashable, Hashable, object]]:
    """Yield edge endpoints (node ids) and payloads.

    Yields
    ------
    tuple[Hashable, Hashable, object]
        Node id endpoints and payloads.
    """
    for src_idx, dst_idx, payload in iter_edge_payloads(store):
        yield store.index_to_id[src_idx], store.index_to_id[dst_idx], payload


def iter_incident_edge_id_payloads(
    store: RxGraphStore,
    node_id: Hashable,
) -> Iterable[tuple[Hashable, Hashable, object]]:
    """Yield incident edge endpoints (node ids) and payloads.

    Yields
    ------
    tuple[Hashable, Hashable, object]
        Node id endpoints and payloads for incident edges.
    """
    node_idx = store.id_to_index.get(node_id)
    if node_idx is None:
        return
    edge_list = store.graph.edge_list()
    payloads = store.graph.edges()
    for edge_idx in store.graph.incident_edges(node_idx):
        src_idx, dst_idx = edge_list[edge_idx]
        yield store.index_to_id[src_idx], store.index_to_id[dst_idx], payloads[edge_idx]


def iter_edge_weights(
    store: RxGraphStore,
    *,
    nan_policy: NanPolicy | None = None,
) -> Iterable[tuple[int, int, float]]:
    """Yield edge endpoints and normalized weights.

    Yields
    ------
    tuple[int, int, float]
        Edge endpoints and normalized weights.
    """
    resolved_nan_policy = nan_policy or store.numeric_policy.nan_policy
    for src_idx, dst_idx, payload in iter_edge_payloads(store):
        weight = edge_weight_from_payload(payload, nan_policy=resolved_nan_policy)
        yield src_idx, dst_idx, weight


def iter_edge_id_weights(
    store: RxGraphStore,
    *,
    nan_policy: NanPolicy | None = None,
) -> Iterable[tuple[Hashable, Hashable, float]]:
    """Yield edge endpoints (node ids) and normalized weights.

    Yields
    ------
    tuple[Hashable, Hashable, float]
        Node id endpoints and normalized weight.
    """
    for src_idx, dst_idx, weight in iter_edge_weights(store, nan_policy=nan_policy):
        yield store.index_to_id[src_idx], store.index_to_id[dst_idx], weight


def iter_incident_edge_id_weights(
    store: RxGraphStore,
    node_id: Hashable,
    *,
    nan_policy: NanPolicy | None = None,
) -> Iterable[tuple[Hashable, Hashable, float]]:
    """Yield incident edge endpoints (node ids) and normalized weights.

    Yields
    ------
    tuple[Hashable, Hashable, float]
        Node id endpoints and normalized weights for incident edges.
    """
    resolved_nan_policy = nan_policy or store.numeric_policy.nan_policy
    for src_id, dst_id, payload in iter_incident_edge_id_payloads(store, node_id):
        weight = edge_weight_from_payload(payload, nan_policy=resolved_nan_policy)
        yield src_id, dst_id, weight


def iter_weighted_edge_ids(
    store: RxGraphStore,
    *,
    nan_policy: NanPolicy | None = None,
) -> Iterable[tuple[Hashable, Hashable, float]]:
    """Yield edge endpoints (node ids) and normalized weights.

    Yields
    ------
    tuple[Hashable, Hashable, float]
        Node id endpoints and normalized weight.
    """
    for src_idx, dst_idx, weight in iter_edge_weights(store, nan_policy=nan_policy):
        yield store.index_to_id[src_idx], store.index_to_id[dst_idx], weight


def edge_weight_map(
    store: RxGraphStore,
    *,
    nan_policy: NanPolicy | None = None,
) -> dict[tuple[int, int], float]:
    """Return a mapping of edge endpoint indices to weights.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping of edge endpoints to normalized weights.
    """
    weights: dict[tuple[int, int], float] = {}
    for src_idx, dst_idx, weight in iter_edge_weights(store, nan_policy=nan_policy):
        if store.is_directed:
            key = (src_idx, dst_idx)
        else:
            key = (min(src_idx, dst_idx), max(src_idx, dst_idx))
        weights[key] = weight
    return weights


def neighbors_by_index(
    store: RxGraphStore,
    *,
    include_self: bool = False,
) -> dict[int, list[int]]:
    """Return deterministic neighbor lists keyed by node index.

    Returns
    -------
    dict[int, list[int]]
        Neighbor lists keyed by node index.
    """
    neighbors: dict[int, set[int]] = {idx: set() for idx in store.graph.node_indices()}
    for src_idx, dst_idx in store.graph.edge_list():
        if src_idx == dst_idx:
            if include_self:
                neighbors[src_idx].add(src_idx)
            continue
        neighbors[src_idx].add(dst_idx)
        if not store.is_directed:
            neighbors[dst_idx].add(src_idx)
    return {
        idx: sorted(values, key=lambda node_idx: stable_key(store.index_to_id[node_idx]))
        for idx, values in neighbors.items()
    }


def weighted_neighbors_by_index(
    store: RxGraphStore,
    edge_weights: Mapping[tuple[int, int], float],
) -> dict[int, list[tuple[int, float]]]:
    """Return weighted neighbor lists keyed by node index.

    Returns
    -------
    dict[int, list[tuple[int, float]]]
        Weighted neighbor lists keyed by node index.
    """
    neighbors: dict[int, list[tuple[int, float]]] = {idx: [] for idx in store.graph.node_indices()}
    for src_idx, dst_idx in store.graph.edge_list():
        key = (
            (src_idx, dst_idx)
            if store.is_directed
            else (min(src_idx, dst_idx), max(src_idx, dst_idx))
        )
        weight = edge_weights.get(key, 1.0)
        if src_idx != dst_idx:
            neighbors[src_idx].append((dst_idx, weight))
            if not store.is_directed:
                neighbors[dst_idx].append((src_idx, weight))
    for idx, items in neighbors.items():
        neighbors[idx] = sorted(items, key=lambda item: stable_key(store.index_to_id[item[0]]))
    return neighbors


__all__ = [
    "edge_index_map",
    "edge_weight_map",
    "iter_edge_id_payloads",
    "iter_edge_id_weights",
    "iter_edge_index_payloads",
    "iter_edge_payloads",
    "iter_edge_weights",
    "iter_incident_edge_id_payloads",
    "iter_incident_edge_id_weights",
    "iter_weighted_edge_ids",
    "neighbors_by_index",
    "weighted_neighbors_by_index",
]
