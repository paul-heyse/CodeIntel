"""Normalization utilities for Arrow arrays and tables."""

from __future__ import annotations

import contextlib

import pyarrow as pa


def normalize_array(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Return a single-chunk array suitable for compute kernels.

    Parameters
    ----------
    values
        Input Arrow array or chunked array.

    Returns
    -------
    normalized : pyarrow.Array
        Normalized array.

    Raises
    ------
    TypeError
        Raised when the input is not an Arrow array.
    """
    if isinstance(values, pa.Array):
        return values
    if isinstance(values, pa.ChunkedArray):
        if values.num_chunks == 0:
            return pa.array([], type=values.type)
        try:
            return values.combine_chunks()
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
            chunks = list(values.iterchunks())
            if not chunks:
                return pa.array([], type=values.type)
            return pa.concat_arrays(chunks)
    msg = "Expected Arrow array values."
    raise TypeError(msg)


def normalize_table(table: pa.Table) -> pa.Table:
    """Unify dictionaries and combine chunks for compute-heavy operations.

    Parameters
    ----------
    table
        Input Arrow table.

    Returns
    -------
    normalized : pyarrow.Table
        Normalized table with unified dictionaries and combined chunks.
    """
    unify = getattr(table, "unify_dictionaries", None)
    if callable(unify):
        with contextlib.suppress(pa.ArrowInvalid):
            table = unify()
    combine = getattr(table, "combine_chunks", None)
    if callable(combine):
        with contextlib.suppress(pa.ArrowInvalid):
            table = combine()
    return table


def normalize_array_for_compute(values: pa.Array | pa.ChunkedArray) -> pa.Array:
    """Return a normalized array for compute-heavy kernels.

    Returns
    -------
    pyarrow.Array
        Normalized array for compute kernels.
    """
    return normalize_array(values)


def normalize_table_for_compute(table: pa.Table) -> pa.Table:
    """Normalize a table for compute-heavy kernels.

    Returns
    -------
    pyarrow.Table
        Normalized table with unified dictionaries and combined chunks.
    """
    return normalize_table(table)


__all__ = [
    "normalize_array",
    "normalize_array_for_compute",
    "normalize_table",
    "normalize_table_for_compute",
]
