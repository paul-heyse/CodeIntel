"""Node payload encoding helpers for rustworkx graphs."""

from __future__ import annotations

import base64
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

from codeintel.build.graphs.rx.normalize import stable_key

_ENCODED_TYPE_KEY = "__rx_type__"
_ENCODED_ITEMS_KEY = "items"
_ENCODED_VALUE_KEY = "value"
_ENCODED_DATA_KEY = "data"
_PAYLOAD_ID_KEY = "id"
_PAYLOAD_ATTRS_KEY = "attrs"
_EDGE_PAYLOAD_WEIGHT_KEY = "weight"
_EDGE_PAYLOAD_METRICS_KEY = "metrics"
_CALLSITE_LEN = 3
NODE_PAYLOAD_VERSION = "v1"
EDGE_PAYLOAD_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class GraphNodePayload:
    """Structured node payload for rustworkx graphs."""

    node_id: Hashable
    node_kind: str | None = None
    label: str | None = None
    path: str | None = None
    span: tuple[int, int, int, int] | None = None
    namespace: str | None = None
    module: str | None = None
    symbol_kind: str | None = None
    block_kind: str | None = None
    synthetic: bool | None = None
    metrics: Mapping[str, float | int | bool] | None = None

    def as_attrs(self) -> dict[str, object]:
        """Return payload attributes as a JSON-friendly mapping.

        Returns
        -------
        dict[str, object]
            Attribute mapping for serialization.
        """
        attrs: dict[str, object | None] = {
            "node_kind": self.node_kind,
            "label": self.label,
            "path": self.path,
            "span": self.span,
            "namespace": self.namespace,
            "module": self.module,
            "symbol_kind": self.symbol_kind,
            "block_kind": self.block_kind,
            "synthetic": self.synthetic,
        }
        if self.metrics is not None:
            attrs["metrics"] = dict(self.metrics)
        return _filter_none(attrs)


@dataclass(frozen=True, slots=True)
class GraphEdgePayload:
    """Structured edge payload for rustworkx graphs."""

    weight: float
    edge_kind: str | None = None
    count: int | None = None
    callsite: tuple[str, int, int] | None = None
    symbol_ref: str | None = None
    config_key: str | None = None
    synthetic: bool | None = None
    metrics: Mapping[str, float | int | bool] | None = None

    def as_attrs(self) -> dict[str, object]:
        """Return payload attributes as a JSON-friendly mapping.

        Returns
        -------
        dict[str, object]
            Attribute mapping for serialization.
        """
        attrs: dict[str, object | None] = {
            _EDGE_PAYLOAD_WEIGHT_KEY: self.weight,
            "edge_kind": self.edge_kind,
            "count": self.count,
            "callsite": self.callsite,
            "symbol_ref": self.symbol_ref,
            "config_key": self.config_key,
            "synthetic": self.synthetic,
        }
        if self.metrics is not None:
            attrs[_EDGE_PAYLOAD_METRICS_KEY] = dict(self.metrics)
        return _filter_none(attrs)


def _is_json_primitive(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def encode_node_id(node_id: Hashable) -> object:
    """Encode node IDs into JSON-safe representations.

    Returns
    -------
    object
        JSON-safe representation of the node ID.
    """
    if _is_json_primitive(node_id):
        return node_id
    if isinstance(node_id, bytes):
        encoded = base64.b64encode(node_id).decode("ascii")
        return {_ENCODED_TYPE_KEY: "bytes", _ENCODED_DATA_KEY: encoded}
    if isinstance(node_id, bytearray):
        encoded = base64.b64encode(bytes(node_id)).decode("ascii")
        return {_ENCODED_TYPE_KEY: "bytes", _ENCODED_DATA_KEY: encoded}
    if isinstance(node_id, tuple):
        items = [encode_node_id(item) for item in node_id]
        return {_ENCODED_TYPE_KEY: "tuple", _ENCODED_ITEMS_KEY: items}
    if isinstance(node_id, frozenset):
        items = sorted(node_id, key=stable_key)
        encoded = [encode_node_id(item) for item in items]
        return {_ENCODED_TYPE_KEY: "frozenset", _ENCODED_ITEMS_KEY: encoded}
    return {_ENCODED_TYPE_KEY: "repr", _ENCODED_VALUE_KEY: repr(node_id)}


def _decode_items(payload: dict[str, object]) -> list[Hashable]:
    raw_items = payload.get(_ENCODED_ITEMS_KEY)
    if not isinstance(raw_items, list):
        return []
    return [decode_node_id(item) for item in raw_items]


def _decode_tuple_payload(payload: dict[str, object]) -> Hashable:
    return tuple(_decode_items(payload))


def _decode_frozenset_payload(payload: dict[str, object]) -> Hashable:
    return frozenset(_decode_items(payload))


def _decode_bytes_payload(payload: dict[str, object]) -> Hashable:
    encoded = payload.get(_ENCODED_DATA_KEY)
    if isinstance(encoded, str):
        return base64.b64decode(encoded.encode("ascii"))
    return b""


def _decode_repr_payload(payload: dict[str, object]) -> Hashable:
    value = payload.get(_ENCODED_VALUE_KEY, "")
    return str(value)


_TAG_DECODERS: dict[str, Callable[[dict[str, object]], Hashable]] = {
    "tuple": _decode_tuple_payload,
    "frozenset": _decode_frozenset_payload,
    "bytes": _decode_bytes_payload,
    "repr": _decode_repr_payload,
}


def _decode_tagged_payload(payload: dict[str, object]) -> Hashable | None:
    tag = payload.get(_ENCODED_TYPE_KEY)
    if not isinstance(tag, str):
        return None
    decoder = _TAG_DECODERS.get(tag)
    if decoder is None:
        return None
    return decoder(payload)


def decode_node_id(payload: object) -> Hashable:
    """Decode node IDs from JSON-safe representations.

    Returns
    -------
    Hashable
        Decoded node identifier.
    """
    if isinstance(payload, dict):
        decoded = _decode_tagged_payload(payload)
        return decoded if decoded is not None else str(payload)

    if isinstance(payload, (list, tuple)):
        decoded = tuple(decode_node_id(item) for item in payload)
    elif isinstance(payload, frozenset):
        decoded = frozenset(decode_node_id(item) for item in payload)
    elif isinstance(payload, (bytes, bytearray)):
        decoded = bytes(payload)
    elif _is_json_primitive(payload):
        decoded = payload
    else:
        decoded = str(payload)
    return decoded


def encode_node_payload(
    node_id: Hashable | GraphNodePayload,
    attrs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Encode node ID and attributes into a rustworkx payload.

    Returns
    -------
    dict[str, object]
        Encoded payload containing node ID and attributes.
    """
    merged_attrs: dict[str, object] = {}
    if isinstance(node_id, GraphNodePayload):
        resolved_id = node_id.node_id
        merged_attrs.update(node_id.as_attrs())
    else:
        resolved_id = node_id
    if attrs:
        merged_attrs.update(dict(attrs))
    payload: dict[str, object] = {_PAYLOAD_ID_KEY: encode_node_id(resolved_id)}
    if merged_attrs:
        payload[_PAYLOAD_ATTRS_KEY] = merged_attrs
    return payload


def decode_node_payload(payload: object) -> tuple[Hashable, dict[str, object]]:
    """Decode a rustworkx node payload into ID and attributes.

    Returns
    -------
    tuple[Hashable, dict[str, object]]
        Node ID and decoded attributes mapping.
    """
    if isinstance(payload, dict) and _PAYLOAD_ID_KEY in payload:
        node_id = decode_node_id(payload.get(_PAYLOAD_ID_KEY))
        attrs_raw = payload.get(_PAYLOAD_ATTRS_KEY)
        if isinstance(attrs_raw, Mapping):
            return node_id, dict(attrs_raw)
        return node_id, {}
    return decode_node_id(payload), {}


def encode_edge_payload(
    payload: GraphEdgePayload | Mapping[str, object],
) -> dict[str, object]:
    """Encode edge payloads into a JSON-friendly mapping.

    Returns
    -------
    dict[str, object]
        Encoded payload mapping with weight and metadata fields.
    """
    if isinstance(payload, GraphEdgePayload):
        return payload.as_attrs()
    return dict(payload)


def decode_edge_payload(payload: object) -> GraphEdgePayload | None:
    """Decode a rustworkx edge payload into a structured payload object.

    Returns
    -------
    GraphEdgePayload | None
        Structured payload when decoding succeeds.
    """
    if payload is None:
        return None
    if isinstance(payload, GraphEdgePayload):
        return payload
    if isinstance(payload, Mapping):
        weight = _coerce_float(payload.get(_EDGE_PAYLOAD_WEIGHT_KEY))
        if weight is None:
            weight = 1.0
        return GraphEdgePayload(
            weight=weight,
            edge_kind=_coerce_str(payload.get("edge_kind")),
            count=_coerce_int(payload.get("count")),
            callsite=_coerce_callsite(payload.get("callsite")),
            symbol_ref=_coerce_str(payload.get("symbol_ref")),
            config_key=_coerce_str(payload.get("config_key")),
            synthetic=_coerce_bool(payload.get("synthetic")),
            metrics=_coerce_metrics(payload.get(_EDGE_PAYLOAD_METRICS_KEY)),
        )
    weight = _coerce_float(payload)
    if weight is None:
        return None
    return GraphEdgePayload(weight=weight)


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_callsite(value: object) -> tuple[str, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != _CALLSITE_LEN:
        return None
    path, line, col = value
    if not isinstance(path, str):
        return None
    if not isinstance(line, int) or not isinstance(col, int):
        return None
    return (path, line, col)


def _coerce_metrics(value: object) -> dict[str, float | int | bool] | None:
    if not isinstance(value, Mapping):
        return None
    metrics: dict[str, float | int | bool] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, (bool, int, float)):
            metrics[key] = item
    if not metrics:
        return None
    return metrics


def _filter_none(attrs: Mapping[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in attrs.items() if value is not None}


__all__ = [
    "EDGE_PAYLOAD_VERSION",
    "NODE_PAYLOAD_VERSION",
    "GraphEdgePayload",
    "GraphNodePayload",
    "decode_edge_payload",
    "decode_node_id",
    "decode_node_payload",
    "encode_edge_payload",
    "encode_node_id",
    "encode_node_payload",
]
