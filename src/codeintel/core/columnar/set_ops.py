"""Set lookup helpers for Arrow arrays."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, cast_array, require_array
from codeintel.core.columnar.normalization import normalize_array
from codeintel.core.columnar.type_normalization import (
    is_binary_view_type,
    normalize_string_view_array,
)


def value_set(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Return a normalized value set array.

    Parameters
    ----------
    values
        Input Arrow array or chunked array.

    Returns
    -------
    normalized : pyarrow.Array
        Normalized value set.
    """
    return normalize_array(values)


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
        resolved = normalize_array(value_set)
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


def is_in_mask(
    values: pa.Array | pa.ChunkedArray,
    *,
    target: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return a mask indicating whether values are in the target set.

    Parameters
    ----------
    values
        Input Arrow array to test.
    target
        Target set of values.

    Returns
    -------
    mask : pyarrow.Array | pyarrow.ChunkedArray
        Boolean membership mask.
    """
    options = pc.SetLookupOptions(value_set=value_set(target))
    result = call_compute("is_in", [values], options=options)
    return require_array(result, name="is_in")


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
    result = call_compute("index_in", [normalized], options=options)
    return require_array(result, name="index_in")


__all__ = [
    "index_in",
    "is_in_mask",
    "value_set",
    "value_set_array",
]
