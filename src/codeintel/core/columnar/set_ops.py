"""Set lookup helpers for Arrow arrays."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array
from codeintel.core.columnar.normalization import normalize_array


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


__all__ = [
    "is_in_mask",
    "value_set",
]
