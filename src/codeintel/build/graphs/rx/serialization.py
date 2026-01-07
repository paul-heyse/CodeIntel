"""Rustworkx node-link serialization helpers."""

from __future__ import annotations

from pathlib import Path

import rustworkx as rx

from codeintel.build.graphs.rx.metadata import metadata_from_graph

RxGraph = rx.PyGraph | rx.PyDiGraph


def dumps_node_link_json(graph: RxGraph, *, require_metadata: bool = False) -> str:
    """Serialize a graph to a node-link JSON string.

    Returns
    -------
    str
        Node-link JSON string for the provided graph.

    Raises
    ------
    ValueError
        If the graph cannot be serialized to node-link JSON.
    """
    if require_metadata and metadata_from_graph(graph) is None:
        message = "Graph metadata missing; refusing to serialize cache payload"
        raise ValueError(message)
    payload = rx.node_link_json(graph)
    if payload is None:
        message = "rustworkx node_link_json returned None"
        raise ValueError(message)
    return payload


def loads_node_link_json(payload: str) -> RxGraph:
    """Parse a node-link JSON payload into a rustworkx graph.

    Returns
    -------
    RxGraph
        Parsed rustworkx graph representation.
    """
    return rx.parse_node_link_json(payload)


def write_node_link_json(path: Path, graph: RxGraph) -> None:
    """Write a node-link JSON payload to disk."""
    payload = dumps_node_link_json(graph)
    path.write_text(payload, encoding="utf-8")


def read_node_link_json(path: Path) -> RxGraph:
    """Read a node-link JSON payload from disk.

    Returns
    -------
    RxGraph
        Parsed rustworkx graph representation.
    """
    payload = path.read_text(encoding="utf-8")
    return loads_node_link_json(payload)


__all__ = [
    "RxGraph",
    "dumps_node_link_json",
    "loads_node_link_json",
    "read_node_link_json",
    "write_node_link_json",
]
