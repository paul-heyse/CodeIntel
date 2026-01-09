"""Rustworkx node-link serialization helpers."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import rustworkx as rx

from codeintel.build.graphs.rx.metadata import metadata_from_graph
from codeintel.core.serialization.payload import PayloadValue, decode_payload

RxGraph = rx.PyGraph | rx.PyDiGraph

_LEGACY_DATA_KEY = "ci_b64_msgpack"
_PAYLOAD_KEY = "payload"


def _coerce_payload_value(value: object | None) -> PayloadValue | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _coerce_payload_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [_coerce_payload_value(item) for item in value]
    return str(value)


def _encode_payload_json(value: object | None) -> str | None:
    if value is None:
        return None
    sanitized = _coerce_payload_value(value)
    try:
        return json.dumps(
            sanitized,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return json.dumps(
            str(sanitized),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def _pack_payload(value: object | None) -> dict[str, str]:
    encoded = _encode_payload_json(value)
    if encoded is None:
        return {}
    return {_PAYLOAD_KEY: encoded}


def _unpack_payload(data: Mapping[str, object] | None) -> object | None:
    if not data:
        return None
    payload = data.get(_PAYLOAD_KEY)
    result: object | None
    if isinstance(payload, str):
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            result = payload
    else:
        raw = data.get(_LEGACY_DATA_KEY)
        if isinstance(raw, str):
            if not raw:
                result = None
            else:
                try:
                    decoded = base64.b64decode(raw.encode("ascii"))
                except ValueError:
                    result = dict(data)
                else:
                    result = decode_payload(decoded)
        else:
            result = dict(data)
    return result


def _graph_attrs_in(attrs: Mapping[str, object] | None) -> dict[str, str]:
    if not attrs:
        return {}
    encoded: dict[str, str] = {}
    for key, value in attrs.items():
        serialized = _encode_payload_json(value)
        if serialized is not None:
            encoded[str(key)] = serialized
    return encoded


def _graph_attrs_out(data: Mapping[str, object] | None) -> dict[str, object]:
    if not data:
        return {}
    decoded: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, str):
            try:
                decoded_value = json.loads(value)
            except json.JSONDecodeError:
                decoded_value = value
        else:
            decoded_value = value
        decoded[str(key)] = decoded_value
    return decoded


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
    if isinstance(graph, rx.PyDiGraph):
        payload = rx.digraph_node_link_json(
            graph,
            graph_attrs=_graph_attrs_in,
            node_attrs=_pack_payload,
            edge_attrs=_pack_payload,
        )
    else:
        payload = rx.graph_node_link_json(
            graph,
            graph_attrs=_graph_attrs_in,
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
