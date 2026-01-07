"""Node payload encoding helpers for rustworkx graphs."""

from __future__ import annotations

import base64
from collections.abc import Callable, Hashable, Mapping

from codeintel.build.graphs.rx.normalize import stable_key

_ENCODED_TYPE_KEY = "__rx_type__"
_ENCODED_ITEMS_KEY = "items"
_ENCODED_VALUE_KEY = "value"
_ENCODED_DATA_KEY = "data"
_PAYLOAD_ID_KEY = "id"
_PAYLOAD_ATTRS_KEY = "attrs"


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
    node_id: Hashable,
    attrs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Encode node ID and attributes into a rustworkx payload.

    Returns
    -------
    dict[str, object]
        Encoded payload containing node ID and attributes.
    """
    payload: dict[str, object] = {_PAYLOAD_ID_KEY: encode_node_id(node_id)}
    if attrs:
        payload[_PAYLOAD_ATTRS_KEY] = dict(attrs)
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


__all__ = [
    "decode_node_id",
    "decode_node_payload",
    "encode_node_id",
    "encode_node_payload",
]
