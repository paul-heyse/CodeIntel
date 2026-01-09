"""Bulk graph construction helpers for rustworkx stores."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.payloads import encode_node_payload
from codeintel.build.graphs.rx.policies import GraphNumericPolicy, GraphWeightPolicy
from codeintel.build.graphs.rx.store import RxGraphStore

DEFAULT_BULK_FLUSH = 250_000
EDGE_WEIGHT_INDEX = 2
MIN_EDGE_FIELDS = 2


def _identity(value: object) -> Hashable | None:
    return value if isinstance(value, Hashable) else None


def _merge_node_ids(
    node_ids: Iterable[Hashable] | None,
    node_attrs: Mapping[Hashable, Mapping[str, object]] | None,
) -> list[Hashable]:
    merged: list[Hashable] = []
    if node_ids is not None:
        merged.extend(node_ids)
    if node_attrs:
        merged.extend(node_attrs.keys())
    return merged


def _dedupe_nodes(node_ids: Iterable[Hashable], *, stable_nodes: bool) -> list[Hashable]:
    seen: set[Hashable] = set()
    ordered: list[Hashable] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
    if stable_nodes:
        ordered.sort(key=stable_key)
    return ordered


def _add_nodes(
    store: RxGraphStore,
    node_ids: Sequence[Hashable],
    node_attrs: Mapping[Hashable, Mapping[str, object]] | None,
) -> None:
    payloads: list[object] = []
    for node_id in node_ids:
        attrs = dict(node_attrs.get(node_id, {})) if node_attrs else {}
        store.node_attrs[node_id] = attrs
        payloads.append(encode_node_payload(node_id, attrs))
    indices = store.graph.add_nodes_from(payloads)
    for node_id, idx in zip(node_ids, indices, strict=True):
        store.id_to_index[node_id] = idx
        store.index_to_id[idx] = node_id
    store.touch()


@dataclass(frozen=True, slots=True)
class EdgeBuildSpec:
    """Configuration for bulk edge ingestion."""

    directed: bool
    weight_policy: GraphWeightPolicy
    numeric_policy: GraphNumericPolicy
    src_fn: Callable[[object], Hashable | None] = _identity
    dst_fn: Callable[[object], Hashable | None] = _identity
    weight_fn: Callable[[object], float] | None = None
    node_attrs_fn: Callable[[Hashable, str], Mapping[str, object]] | None = None


@dataclass(frozen=True, slots=True)
class BuildStoreOptions:
    """Options for bulk store construction."""

    stable_nodes: bool = True
    aggregate_edges: bool = True
    flush_every: int = DEFAULT_BULK_FLUSH
    node_ids: Iterable[Hashable] | None = None
    node_attrs: Mapping[Hashable, Mapping[str, object]] | None = None
    node_hint: int | None = None
    edge_hint: int | None = None


@dataclass(slots=True)
class BulkEdgeInserter:
    """Incremental bulk edge inserter with aggregation."""

    store: RxGraphStore
    flush_every: int = DEFAULT_BULK_FLUSH
    stable_nodes: bool = True
    aggregate_edges: bool = True
    _edges: dict[tuple[Hashable, Hashable], float] = field(default_factory=dict)
    _touched_nodes: set[Hashable] = field(default_factory=set)

    def add(
        self,
        src_id: Hashable,
        dst_id: Hashable,
        *,
        weight: object | None,
        src_attrs: Mapping[str, object] | None = None,
        dst_attrs: Mapping[str, object] | None = None,
    ) -> None:
        """Register an edge payload for later bulk insertion."""
        self._touched_nodes.add(src_id)
        self._touched_nodes.add(dst_id)
        if src_attrs:
            self._merge_node_attrs(src_id, src_attrs)
        if dst_attrs:
            self._merge_node_attrs(dst_id, dst_attrs)
        normalized = self.store.weight_policy.normalize_weight(weight)
        key = _edge_key(src_id, dst_id, directed=self.store.is_directed)
        if not self.aggregate_edges:
            self._edges[key] = normalized
        else:
            current = self._edges.get(key)
            if current is None:
                self._edges[key] = normalized
            else:
                self._edges[key] = self.store.weight_policy.combine_weights(current, normalized)
        if len(self._edges) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Flush buffered nodes and edges to the rustworkx graph."""
        if not self._edges:
            self._touched_nodes.clear()
            return
        self._ensure_nodes()
        edge_triples = [
            (
                self.store.id_to_index[src_id],
                self.store.id_to_index[dst_id],
                weight,
            )
            for (src_id, dst_id), weight in self._edges.items()
        ]
        self.store.graph.add_edges_from(edge_triples)
        self.store.touch()
        self._edges.clear()
        self._touched_nodes.clear()

    def _ensure_nodes(self) -> None:
        new_nodes = [
            node_id for node_id in self._touched_nodes if node_id not in self.store.id_to_index
        ]
        if not new_nodes:
            return
        if self.stable_nodes:
            new_nodes.sort(key=stable_key)
        payloads: list[object] = []
        for node_id in new_nodes:
            attrs = self.store.node_attrs.get(node_id, {})
            payloads.append(encode_node_payload(node_id, attrs))
        indices = self.store.graph.add_nodes_from(payloads)
        for node_id, idx in zip(new_nodes, indices, strict=True):
            self.store.id_to_index[node_id] = idx
            self.store.index_to_id[idx] = node_id

    def _merge_node_attrs(self, node_id: Hashable, attrs: Mapping[str, object]) -> None:
        existing = self.store.node_attrs.setdefault(node_id, {})
        existing.update(dict(attrs))


def build_store_from_edge_tuples(
    edges: Iterable[Sequence[object]],
    *,
    spec: EdgeBuildSpec,
    options: BuildStoreOptions | None = None,
) -> RxGraphStore:
    """Build an RxGraphStore from edge tuples with bulk insertion.

    Returns
    -------
    RxGraphStore
        Graph store populated from the provided edge tuples.
    """
    resolved = options or BuildStoreOptions()
    merged_node_ids = _merge_node_ids(resolved.node_ids, resolved.node_attrs)
    resolved_node_ids = (
        _dedupe_nodes(merged_node_ids, stable_nodes=resolved.stable_nodes)
        if merged_node_ids
        else []
    )
    resolved_node_hint = resolved.node_hint
    if resolved_node_hint is None and resolved_node_ids:
        resolved_node_hint = len(resolved_node_ids)
    store = (
        RxGraphStore.directed(
            node_hint=resolved_node_hint,
            edge_hint=resolved.edge_hint,
            weight_policy=spec.weight_policy,
            numeric_policy=spec.numeric_policy,
        )
        if spec.directed
        else RxGraphStore.undirected(
            node_hint=resolved_node_hint,
            edge_hint=resolved.edge_hint,
            weight_policy=spec.weight_policy,
            numeric_policy=spec.numeric_policy,
        )
    )
    if resolved_node_ids:
        _add_nodes(store, resolved_node_ids, resolved.node_attrs)
    inserter = BulkEdgeInserter(
        store=store,
        flush_every=resolved.flush_every,
        stable_nodes=resolved.stable_nodes,
        aggregate_edges=resolved.aggregate_edges,
    )
    for row in edges:
        if len(row) < MIN_EDGE_FIELDS:
            continue
        src_id = spec.src_fn(row[0])
        dst_id = spec.dst_fn(row[1])
        if src_id is None or dst_id is None:
            continue
        raw_weight = row[EDGE_WEIGHT_INDEX] if len(row) > EDGE_WEIGHT_INDEX else None
        weight = spec.weight_fn(raw_weight) if spec.weight_fn is not None else raw_weight
        src_attrs = spec.node_attrs_fn(src_id, "src") if spec.node_attrs_fn else None
        dst_attrs = spec.node_attrs_fn(dst_id, "dst") if spec.node_attrs_fn else None
        inserter.add(src_id, dst_id, weight=weight, src_attrs=src_attrs, dst_attrs=dst_attrs)
    inserter.flush()
    return store


def _edge_key(
    src_id: Hashable,
    dst_id: Hashable,
    *,
    directed: bool,
) -> tuple[Hashable, Hashable]:
    if directed:
        return (src_id, dst_id)
    return (src_id, dst_id) if stable_key(src_id) <= stable_key(dst_id) else (dst_id, src_id)


__all__ = [
    "BuildStoreOptions",
    "BulkEdgeInserter",
    "EdgeBuildSpec",
    "build_store_from_edge_tuples",
]
