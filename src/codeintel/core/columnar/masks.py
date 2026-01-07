"""Boolean mask helpers for Arrow compute pipelines."""

from __future__ import annotations

import pyarrow as pa

from codeintel.core.columnar.compute_helpers import call_compute, require_array


def fill_null_false(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Replace nulls in a boolean mask with False.

    Parameters
    ----------
    mask
        Input boolean mask.

    Returns
    -------
    filled : pyarrow.Array | pyarrow.ChunkedArray
        Mask with nulls replaced by False.
    """
    result = call_compute("fill_null", [mask, pa.scalar(value=False)])
    return require_array(result, name="fill_null")


def invert_mask(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Invert a boolean mask.

    Parameters
    ----------
    mask
        Input boolean mask.

    Returns
    -------
    inverted : pyarrow.Array | pyarrow.ChunkedArray
        Inverted boolean mask.
    """
    result = call_compute("invert", [mask])
    return require_array(result, name="invert")


def and_mask(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Return the Kleene AND of two boolean masks.

    Parameters
    ----------
    left
        Left-hand boolean mask.
    right
        Right-hand boolean mask.

    Returns
    -------
    combined : pyarrow.Array | pyarrow.ChunkedArray
        Combined boolean mask.
    """
    result = call_compute("and_kleene", [left, right])
    return require_array(result, name="and_kleene")


def is_valid_mask(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return a boolean mask indicating valid (non-null) entries.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Boolean validity mask.
    """
    result = call_compute("is_valid", [values])
    return require_array(result, name="is_valid")


def filter_valid(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Filter to valid (non-null) entries using Arrow kernels.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Filtered values with nulls removed.
    """
    mask = is_valid_mask(values)
    result = call_compute("filter", [values, mask])
    return require_array(result, name="filter")


__all__ = [
    "and_mask",
    "fill_null_false",
    "filter_valid",
    "invert_mask",
    "is_valid_mask",
]
