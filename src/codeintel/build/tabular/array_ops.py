"""Shared Arrow array helpers for compute-heavy operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.tabular.compute_helpers import cast_array, scalar_from_compute, take_array
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
                resolved = cast_array(resolved, pa.string(), safe=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
                return resolved
        elif is_binary_view_type(like.type):
            try:
                resolved = cast_array(resolved, pa.binary(), safe=True)
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
    *,
    missing_policy: Literal["error", "null"] = "error",
) -> pa.Array | pa.ChunkedArray:
    """Return values aligned to keys via vectorized index lookup.

    Parameters
    ----------
    keys
        Keys to map onto the key_set ordering.
    key_set
        Lookup key set that matches the values array.
    values
        Values aligned to the key_set ordering.
    missing_policy
        Behavior when keys are missing: raise ("error") or return nulls ("null").

    Returns
    -------
    pa.Array | pa.ChunkedArray
        Values aligned to the key order.
    """
    indices = ensure_array(index_in(keys, value_set=key_set))
    missing_mask = pc.less(indices, pa.scalar(0))
    missing_any = scalar_from_compute("any", [missing_mask])
    if missing_policy == "error" and missing_any:
        msg = "take_by_key missing keys"
        raise ValueError(msg)
    safe_indices = pc.if_else(missing_mask, pa.scalar(0), indices)
    selected = take_array(ensure_array(values), safe_indices)
    if missing_policy == "null":
        nulls = pa.nulls(len(indices), type=values.type)
        return pc.if_else(missing_mask, nulls, selected)
    return selected


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
