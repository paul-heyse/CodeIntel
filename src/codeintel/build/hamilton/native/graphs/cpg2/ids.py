"""CPG ID helpers for anchor-map based assembly."""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.build.graphs.assembly.ids import payload_bytes, stable_decimal_id, stable_int_hash

ORDINAL_MOD = 2**31 - 1


def cpg_node_id(table_key: str, pk: Mapping[str, object]) -> int:
    """Return a stable CPG node identifier for a source table row.

    Returns
    -------
    int
        Deterministic DECIMAL(38,0)-safe identifier.
    """
    payload = {"table_key": table_key, "pk": dict(pk)}
    return stable_decimal_id(payload, digest_size=16)


def cpg_source_pk_json(pk: Mapping[str, object]) -> bytes:
    """Encode a primary-key payload for CPG nodes.

    Returns
    -------
    bytes
        Serialized primary key payload.
    """
    return payload_bytes(pk)


def cpg_edge_ordinal(table_key: str, payload: Mapping[str, object]) -> int:
    """Return a stable edge ordinal for deterministic ordering.

    Returns
    -------
    int
        Deterministic ordinal for edge ordering.
    """
    wrapped = {"table_key": table_key, "payload": dict(payload)}
    return stable_int_hash(wrapped, digest_size=8, modulus=ORDINAL_MOD)


__all__ = [
    "ORDINAL_MOD",
    "cpg_edge_ordinal",
    "cpg_node_id",
    "cpg_source_pk_json",
]
