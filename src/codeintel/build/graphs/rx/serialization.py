"""Rustworkx node-link serialization helpers."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path

import rustworkx as rx

from codeintel.build.graphs.rx.metadata import metadata_from_graph
from codeintel.core.serialization.payload import PayloadValue, decode_payload, encode_payload

RxGraph = rx.PyGraph | rx.PyDiGraph

_CI_DATA_KEY = "ci_b64_msgpack"


def _coerce_payload_value(value: object | None) -> PayloadValue | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return list(value)
    return str(value)


def _pack_payload(value: object | None) -> dict[str, str]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = encode_payload(value)
    else:
        raw = encode_payload(_coerce_payload_value(value))
    if raw is None:
        return {}
    encoded = base64.b64encode(raw).decode("ascii")
    return {_CI_DATA_KEY: encoded}


def _unpack_payload(data: Mapping[str, str] | None) -> object | None:
    if not data:
        return None
    raw = data.get(_CI_DATA_KEY)
    if raw is None:
        return dict(data)
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw.encode("ascii"))
    except ValueError:
        return dict(data)
    return decode_payload(decoded)


def _graph_attrs_out(data: Mapping[str, str] | None) -> dict[str, object]:
    unpacked = _unpack_payload(data)
    if isinstance(unpacked, dict):
        return dict(unpacked)
    return {}


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
    payload = rx.node_link_json(
        graph,
        graph_attrs=_pack_payload,
        node_attrs=_pack_payload,
        edge_attrs=_pack_payload,
    )
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
    return rx.parse_node_link_json(
        payload,
        graph_attrs=_graph_attrs_out,
        node_attrs=_unpack_payload,
        edge_attrs=_unpack_payload,
    )


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
