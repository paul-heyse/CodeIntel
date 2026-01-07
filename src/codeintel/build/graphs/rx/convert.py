"""Conversion helpers between rustworkx and NetworkX graphs."""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import rustworkx as rx

from codeintel.build.graphs.rx.normalize import edge_weight_from_payload
from codeintel.build.graphs.rx.payloads import decode_node_payload
from codeintel.build.graphs.rx.store import RxGraphStore

RxGraph = rx.PyGraph | rx.PyDiGraph


def networkx_to_rx(graph: nx.Graph | nx.DiGraph) -> RxGraphStore:
    """Convert a NetworkX graph into a rustworkx-backed store.

    Returns
    -------
    RxGraphStore
        Rustworkx graph store populated from the NetworkX graph.
    """
    store = RxGraphStore.directed() if isinstance(graph, nx.DiGraph) else RxGraphStore.undirected()
    for node_id, attrs in graph.nodes(data=True):
        store.set_node_attrs(node_id, attrs)
    for src_id, dst_id, attrs in graph.edges(data=True):
        weight = edge_weight_from_payload(attrs.get("weight"))
        store.add_weighted_edge(src_id, dst_id, weight=weight)
    return store


def rx_to_networkx(graph: RxGraph) -> nx.Graph:
    """Convert a rustworkx graph into a NetworkX graph with decoded payloads.

    Returns
    -------
    nx.Graph
        NetworkX graph populated from the rustworkx payloads.
    """
    if isinstance(graph, rx.PyDiGraph):
        nx_graph: nx.Graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()

    node_id_by_index: dict[int, Hashable] = {}
    for node_idx in graph.node_indices():
        payload = graph.get_node_data(node_idx)
        node_id, attrs = decode_node_payload(payload)
        node_id_by_index[node_idx] = node_id
        nx_graph.add_node(node_id, **attrs)

    for src_idx, dst_idx in graph.edge_list():
        src_id = node_id_by_index.get(src_idx)
        dst_id = node_id_by_index.get(dst_idx)
        if src_id is None or dst_id is None:
            continue
        payload = graph.get_edge_data(src_idx, dst_idx)
        nx_graph.add_edge(src_id, dst_id, weight=edge_weight_from_payload(payload))
    return nx_graph


__all__ = ["RxGraph", "networkx_to_rx", "rx_to_networkx"]
