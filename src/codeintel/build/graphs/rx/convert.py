"""Conversion helpers between rustworkx graphs and stores."""

from __future__ import annotations

from collections.abc import Hashable

import rustworkx as rx

from codeintel.build.graphs.rx.payloads import decode_node_payload
from codeintel.build.graphs.rx.store import RxGraphStore

RxGraph = rx.PyGraph | rx.PyDiGraph


def store_from_rx(graph: RxGraph) -> RxGraphStore:
    """Convert a rustworkx graph into a rustworkx-backed store.

    Returns
    -------
    RxGraphStore
        Rustworkx graph store populated from the rustworkx payloads.
    """
    id_to_index: dict[Hashable, int] = {}
    index_to_id: dict[int, Hashable] = {}
    node_attrs: dict[Hashable, dict[str, object]] = {}
    for node_idx in graph.node_indices():
        node_id, attrs = decode_node_payload(graph.get_node_data(node_idx))
        id_to_index[node_id] = node_idx
        index_to_id[node_idx] = node_id
        node_attrs[node_id] = attrs
    return RxGraphStore(
        graph=graph,
        id_to_index=id_to_index,
        index_to_id=index_to_id,
        node_attrs=node_attrs,
        is_directed=isinstance(graph, rx.PyDiGraph),
    )


__all__ = ["RxGraph", "store_from_rx"]
