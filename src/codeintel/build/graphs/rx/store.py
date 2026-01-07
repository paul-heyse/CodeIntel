"""Rustworkx graph store with stable ID/index mapping."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import rustworkx as rx

from codeintel.build.graphs.rx.normalize import edge_weight_from_payload, stable_key

RxGraph = rx.PyGraph[Hashable, float] | rx.PyDiGraph[Hashable, float]


@dataclass(slots=True)
class RxGraphStore:
    """Graph store that maintains a stable mapping between IDs and node indices."""

    graph: RxGraph
    id_to_index: dict[Hashable, int]
    index_to_id: dict[int, Hashable]
    is_directed: bool

    @classmethod
    def directed(
        cls,
        *,
        node_hint: int | None = None,
        edge_hint: int | None = None,
    ) -> RxGraphStore:
        """Build a directed store with optional capacity hints.

        Returns
        -------
        RxGraphStore
            Directed graph store instance.
        """
        graph = rx.PyDiGraph(
            multigraph=False,
            node_count_hint=node_hint,
            edge_count_hint=edge_hint,
        )
        return cls(graph=graph, id_to_index={}, index_to_id={}, is_directed=True)

    @classmethod
    def undirected(
        cls,
        *,
        node_hint: int | None = None,
        edge_hint: int | None = None,
    ) -> RxGraphStore:
        """Build an undirected store with optional capacity hints.

        Returns
        -------
        RxGraphStore
            Undirected graph store instance.
        """
        graph = rx.PyGraph(
            multigraph=False,
            node_count_hint=node_hint,
            edge_count_hint=edge_hint,
        )
        return cls(graph=graph, id_to_index={}, index_to_id={}, is_directed=False)

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
        index = self.graph.add_node(node_id)
        self.id_to_index[node_id] = index
        self.index_to_id[index] = node_id
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
        weight = edge_weight_from_payload(payload)
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
        increment = float(weight)
        if self.graph.has_edge(src_idx, dst_idx):
            current_payload = self.graph.get_edge_data(src_idx, dst_idx)
            current = edge_weight_from_payload(current_payload)
            self.graph.update_edge(src_idx, dst_idx, current + increment)
            return
        self.graph.add_edge(src_idx, dst_idx, increment)


__all__ = ["RxGraph", "RxGraphStore"]
