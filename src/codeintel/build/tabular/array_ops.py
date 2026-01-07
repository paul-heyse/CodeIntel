"""Shared Arrow array helpers for compute-heavy operations."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar import normalization as _core_normalization
from codeintel.core.columnar import type_normalization as _type_normalization

is_binary_view_type = _type_normalization.is_binary_view_type
normalize_binary_view_array = _type_normalization.normalize_binary_view_array
normalize_string_view_array = _type_normalization.normalize_string_view_array
normalize_array_for_compute = _core_normalization.normalize_array_for_compute
normalize_table_for_compute = _core_normalization.normalize_table_for_compute


def ensure_array(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Return a contiguous Arrow array for compute kernels.

    Returns
    -------
    pa.Array
        Contiguous Arrow array.
    """
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    return values


def value_set_array(
    value_set: Sequence[object] | pa.Array | pa.ChunkedArray,
    *,
    like: pa.Array | pa.ChunkedArray | None = None,
) -> pa.Array:
    """Normalize a value set into an Arrow array for set-lookup kernels.

    Returns
    -------
    pa.Array
        Arrow array containing the value set.
    """
    if isinstance(value_set, (pa.Array, pa.ChunkedArray)):
        resolved = ensure_array(value_set)
    elif isinstance(value_set, (str, bytes, bytearray)):
        resolved = pa.array([value_set])
    else:
        resolved = pa.array(list(value_set))
    if like is not None:
        if pa.types.is_string_view(like.type):
            try:
                resolved = pc.cast(resolved, pa.string())
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
                return resolved
        elif is_binary_view_type(like.type):
            try:
                resolved = pc.cast(resolved, pa.binary())
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
                return resolved
    return resolved


def index_in(
    values: pa.Array | pa.ChunkedArray,
    *,
    value_set: Sequence[object] | pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return index positions of values in a lookup set.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Index positions per input value.
    """
    normalized = normalize_string_view_array(values)
    resolved = value_set_array(value_set, like=normalized)
    options = pc.SetLookupOptions(value_set=resolved)
    return pc.call_function("index_in", [normalized], options=options)


def take_by_key(
    keys: pa.Array | pa.ChunkedArray,
    key_set: pa.Array | pa.ChunkedArray,
    values: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return values aligned to keys via vectorized index lookup.

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Values aligned to the key order.
    """
    indices = index_in(keys, value_set=key_set)
    return pc.take(ensure_array(values), indices)


__all__ = [
    "ensure_array",
    "index_in",
    "normalize_array_for_compute",
    "normalize_binary_view_array",
    "normalize_string_view_array",
    "normalize_table_for_compute",
    "take_by_key",
    "value_set_array",
]
