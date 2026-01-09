"""Rustworkx graph store with stable ID/index mapping."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import cast

import rustworkx as rx

from codeintel.build.graphs.rx.metadata import (
    GraphMetadata,
    apply_graph_metadata,
    metadata_from_graph,
)
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.rx.payloads import decode_node_payload, encode_node_payload
from codeintel.build.graphs.rx.policies import (
    DEFAULT_NUMERIC_POLICY,
    DEFAULT_WEIGHT_POLICY,
    GraphNumericPolicy,
    GraphWeightPolicy,
    weight_policy_for_name,
)

RxGraph = rx.PyGraph[object, float] | rx.PyDiGraph[object, float]


@dataclass(slots=True)
class RxGraphStore:
    """Graph store that maintains a stable mapping between IDs and node indices."""

    graph: RxGraph
    id_to_index: dict[Hashable, int]
    index_to_id: dict[int, Hashable]
    node_attrs: dict[Hashable, dict[str, object]]
    is_directed: bool
    weight_policy: GraphWeightPolicy = DEFAULT_WEIGHT_POLICY
    numeric_policy: GraphNumericPolicy = DEFAULT_NUMERIC_POLICY
    _version: int = 0
    _view_cache: dict[str, tuple[int, RxGraphStore]] = field(default_factory=dict, repr=False)

    @classmethod
    def directed(
        cls,
        *,
        node_hint: int | None = None,
        edge_hint: int | None = None,
        weight_policy: GraphWeightPolicy | None = None,
        numeric_policy: GraphNumericPolicy | None = None,
    ) -> RxGraphStore:
        """Build a directed store with optional capacity hints.

        Returns
        -------
        RxGraphStore
            Directed graph store instance.
        """
        resolved_weight_policy = weight_policy or DEFAULT_WEIGHT_POLICY
        resolved_numeric_policy = numeric_policy or DEFAULT_NUMERIC_POLICY
        graph = rx.PyDiGraph(
            multigraph=False,
            node_count_hint=node_hint,
            edge_count_hint=edge_hint,
        )
        apply_graph_metadata(graph, GraphMetadata(weight_policy=resolved_weight_policy.name))
        return cls(
            graph=graph,
            id_to_index={},
            index_to_id={},
            node_attrs={},
            is_directed=True,
            weight_policy=resolved_weight_policy,
            numeric_policy=resolved_numeric_policy,
        )

    @classmethod
    def undirected(
        cls,
        *,
        node_hint: int | None = None,
        edge_hint: int | None = None,
        weight_policy: GraphWeightPolicy | None = None,
        numeric_policy: GraphNumericPolicy | None = None,
    ) -> RxGraphStore:
        """Build an undirected store with optional capacity hints.

        Returns
        -------
        RxGraphStore
            Undirected graph store instance.
        """
        resolved_weight_policy = weight_policy or DEFAULT_WEIGHT_POLICY
        resolved_numeric_policy = numeric_policy or DEFAULT_NUMERIC_POLICY
        graph = rx.PyGraph(
            multigraph=False,
            node_count_hint=node_hint,
            edge_count_hint=edge_hint,
        )
        apply_graph_metadata(graph, GraphMetadata(weight_policy=resolved_weight_policy.name))
        return cls(
            graph=graph,
            id_to_index={},
            index_to_id={},
            node_attrs={},
            is_directed=False,
            weight_policy=resolved_weight_policy,
            numeric_policy=resolved_numeric_policy,
        )

    @classmethod
    def from_rx_graph(
        cls,
        graph: RxGraph,
        *,
        weight_policy: GraphWeightPolicy | None = None,
        numeric_policy: GraphNumericPolicy | None = None,
    ) -> RxGraphStore:
        """Build a store from an existing rustworkx graph.

        Returns
        -------
        RxGraphStore
            Store wrapping the provided rustworkx graph.
        """
        id_to_index: dict[Hashable, int] = {}
        index_to_id: dict[int, Hashable] = {}
        node_attrs: dict[Hashable, dict[str, object]] = {}
        for node_idx in graph.node_indices():
            node_id, attrs = decode_node_payload(graph.get_node_data(node_idx))
            id_to_index[node_id] = node_idx
            index_to_id[node_idx] = node_id
            node_attrs[node_id] = attrs
        resolved_policy = weight_policy
        metadata = metadata_from_graph(graph)
        if resolved_policy is None and metadata is not None:
            resolved_policy = weight_policy_for_name(metadata.weight_policy)
        if resolved_policy is None:
            resolved_policy = DEFAULT_WEIGHT_POLICY
        if metadata is None:
            metadata = GraphMetadata(weight_policy=resolved_policy.name)
        elif metadata.weight_policy != resolved_policy.name:
            metadata = GraphMetadata(
                weight_policy=resolved_policy.name,
                cache_version=metadata.cache_version,
                engine=metadata.engine,
                graph_kind=metadata.graph_kind,
                node_payload_version=metadata.node_payload_version,
                determinism_tier=metadata.determinism_tier,
            )
        apply_graph_metadata(graph, metadata)
        return cls(
            graph=graph,
            id_to_index=id_to_index,
            index_to_id=index_to_id,
            node_attrs=node_attrs,
            is_directed=isinstance(graph, rx.PyDiGraph),
            weight_policy=resolved_policy,
            numeric_policy=numeric_policy or DEFAULT_NUMERIC_POLICY,
        )

    @property
    def version(self) -> int:
        """Return the current mutation version."""
        return self._version

    def touch(self) -> None:
        """Invalidate cached views after mutating the graph."""
        self._touch()

    def _touch(self) -> None:
        self._version += 1
        self._view_cache.clear()

    def add_node(
        self,
        node_id: Hashable,
        *,
        attrs: Mapping[str, object] | None = None,
    ) -> int:
        """Add a node with optional attributes.

        Returns
        -------
        int
            Node index for the added node.
        """
        index = self.ensure_node(node_id)
        if attrs:
            self.set_node_attrs(node_id, attrs)
        return index

    def ensure_node(self, node_id: Hashable) -> int:
        """Return the node index for a domain ID, adding it when missing.

        Returns
        -------
        int
            Node index associated with the domain ID.
        """
        existing = self.id_to_index.get(node_id)
        if existing is not None:
            return existing
        attrs = self.node_attrs.setdefault(node_id, {})
        payload = encode_node_payload(node_id, attrs)
        index = self.graph.add_node(payload)
        self.id_to_index[node_id] = index
        self.index_to_id[index] = node_id
        self._touch()
        return index

    def get_index(self, node_id: Hashable) -> int | None:
        """Return the node index for a domain ID when present.

        Returns
        -------
        int | None
            Node index when present, otherwise None.
        """
        return self.id_to_index.get(node_id)

    def get_id(self, index: int) -> Hashable:
        """Return the domain ID for a node index.

        Returns
        -------
        Hashable
            Domain ID stored at the node index.
        """
        return self.index_to_id[index]

    def get_node_attrs(self, node_id: Hashable) -> dict[str, object]:
        """Return the attribute mapping for a node ID.

        Returns
        -------
        dict[str, object]
            Node attribute mapping, or an empty dict if absent.
        """
        attrs = self.node_attrs.get(node_id)
        if attrs is None:
            return {}
        return dict(attrs)

    def set_node_attrs(self, node_id: Hashable, attrs: Mapping[str, object]) -> None:
        """Merge attributes into the node payload."""
        if not attrs:
            self.ensure_node(node_id)
            return
        current = self.node_attrs.setdefault(node_id, {})
        current.update(attrs)
        index = self.ensure_node(node_id)
        payload = encode_node_payload(node_id, current)
        self.graph[index] = payload
        self._touch()

    def remove_node(self, node_id: Hashable) -> bool:
        """Remove a node and its incident edges when present.

        Returns
        -------
        bool
            True when the node was removed.
        """
        index = self.id_to_index.pop(node_id, None)
        if index is None:
            return False
        self.graph.remove_node(index)
        self.index_to_id.pop(index, None)
        self.node_attrs.pop(node_id, None)
        self._touch()
        return True

    def remove_edge(self, src_id: Hashable, dst_id: Hashable) -> bool:
        """Remove an edge when present.

        Returns
        -------
        bool
            True when the edge was removed.
        """
        src_idx = self.id_to_index.get(src_id)
        dst_idx = self.id_to_index.get(dst_id)
        if src_idx is None or dst_idx is None:
            return False
        if not self.graph.has_edge(src_idx, dst_idx):
            return False
        self.graph.remove_edge(src_idx, dst_idx)
        self._touch()
        return True

    def as_undirected(self) -> RxGraphStore:
        """Return an undirected store representation.

        Returns
        -------
        RxGraphStore
            Undirected store representation.
        """
        if not self.is_directed:
            return self
        cached = self._view_cache.get("undirected")
        if cached is not None and cached[0] == self._version:
            return cached[1]
        directed_graph = cast("rx.PyDiGraph", self.graph)
        undirected = directed_graph.to_undirected()
        store = RxGraphStore.from_rx_graph(
            undirected,
            weight_policy=self.weight_policy,
            numeric_policy=self.numeric_policy,
        )
        self._view_cache["undirected"] = (self._version, store)
        return store

    def as_directed(self) -> RxGraphStore:
        """Return a directed store representation.

        Returns
        -------
        RxGraphStore
            Directed store representation.
        """
        if self.is_directed:
            return self
        cached = self._view_cache.get("directed")
        if cached is not None and cached[0] == self._version:
            return cached[1]
        undirected_graph = cast("rx.PyGraph", self.graph)
        directed = undirected_graph.to_directed()
        store = RxGraphStore.from_rx_graph(
            directed,
            weight_policy=self.weight_policy,
            numeric_policy=self.numeric_policy,
        )
        self._view_cache["directed"] = (self._version, store)
        return store

    def node_ids(self) -> list[Hashable]:
        """Return domain IDs in a deterministic ordering.

        Returns
        -------
        list[Hashable]
            Domain IDs sorted by the stable key function.
        """
        return sorted(self.id_to_index.keys(), key=stable_key)

    def node_indices(self) -> list[int]:
        """Return node indices in ascending order.

        Returns
        -------
        list[int]
            Node indices sorted in ascending order.
        """
        return sorted(self.index_to_id.keys())

    def add_edge(
        self,
        src_id: Hashable,
        dst_id: Hashable,
        payload: object | None = None,
    ) -> None:
        """Add an edge, coercing payloads into weights."""
        weight = self.weight_policy.normalize_weight(payload)
        self.add_weighted_edge(src_id, dst_id, weight=weight)

    def add_weighted_edge(
        self,
        src_id: Hashable,
        dst_id: Hashable,
        *,
        weight: float,
    ) -> None:
        """Add an edge and aggregate weights when the edge already exists."""
        src_idx = self.ensure_node(src_id)
        dst_idx = self.ensure_node(dst_id)
        increment = self.weight_policy.normalize_weight(weight)
        if self.graph.has_edge(src_idx, dst_idx):
            current_payload = self.graph.get_edge_data(src_idx, dst_idx)
            current = self.weight_policy.normalize_weight(current_payload)
            updated = self.weight_policy.combine_weights(current, increment)
            self.graph.update_edge(src_idx, dst_idx, updated)
            self._touch()
            return
        self.graph.add_edge(src_idx, dst_idx, increment)
        self._touch()

    def set_edge_weight(
        self,
        src_id: Hashable,
        dst_id: Hashable,
        *,
        weight: float,
    ) -> bool:
        """Set an edge weight without applying aggregation.

        Returns
        -------
        bool
            True when the edge weight was updated.
        """
        src_idx = self.id_to_index.get(src_id)
        dst_idx = self.id_to_index.get(dst_id)
        if src_idx is None or dst_idx is None:
            return False
        normalized = self.weight_policy.normalize_weight(weight)
        if self.graph.has_edge(src_idx, dst_idx):
            self.graph.update_edge(src_idx, dst_idx, normalized)
        else:
            self.graph.add_edge(src_idx, dst_idx, normalized)
        self._touch()
        return True


__all__ = ["RxGraph", "RxGraphStore"]
