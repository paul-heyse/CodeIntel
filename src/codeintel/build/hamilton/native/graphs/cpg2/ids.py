"""CPG ID helpers for anchor-map based assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from codeintel.build.graphs.assembly.ids import payload_bytes, stable_decimal_id, stable_int_hash
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.kernels import hash_struct_ordinal

ORDINAL_MOD = 2**31 - 1
_ORDINAL_TABLE_KEY = "_cpg_edge_table_key"


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


def cpg_edge_ordinals(
    table: pa.Table,
    *,
    table_key: str,
    columns: Sequence[str],
) -> pa.Array | pa.ChunkedArray:
    """Return deterministic edge ordinals derived from Arrow hash kernels.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Ordinal values derived from hash kernels.
    """
    if table.num_rows == 0:
        return pa.nulls(0, type=pa.int64())
    source = append_constant_columns(table, {_ORDINAL_TABLE_KEY: table_key})
    hash_columns = [_ORDINAL_TABLE_KEY]
    hash_columns.extend([column for column in columns if column in source.column_names])
    return hash_struct_ordinal(source, columns=hash_columns, modulus=ORDINAL_MOD)


__all__ = [
    "ORDINAL_MOD",
    "cpg_edge_ordinal",
    "cpg_edge_ordinals",
    "cpg_node_id",
    "cpg_source_pk_json",
]
