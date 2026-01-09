"""Shared kernel primitives for columnar operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array, sort_options

SortKey = tuple[str, Literal["ascending", "descending"]]


def stable_sort_indices(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    null_placement: Literal["at_end", "at_start"] = "at_end",
) -> pa.Array | pa.ChunkedArray:
    """Return stable sort indices for a table.

    Parameters
    ----------
    table
        Table to sort.
    sort_keys
        Sequence of (column, order) pairs.
    null_placement
        Null placement policy.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Indices that define a stable sort order.

    Raises
    ------
    TypeError
        If Arrow does not return an array.
    """
    options = sort_options(sort_keys, null_placement=null_placement)
    result = call_compute("sort_indices", [table], options=options)
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    msg = "Arrow compute sort_indices did not return an array."
    raise TypeError(msg)


def stable_sort_table(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    null_placement: Literal["at_end", "at_start"] = "at_end",
) -> pa.Table:
    """Return a table sorted using stable sort indices.

    Parameters
    ----------
    table
        Table to sort.
    sort_keys
        Sequence of (column, order) pairs.
    null_placement
        Null placement policy.

    Returns
    -------
    pyarrow.Table
        Table sorted by the provided keys.
    """
    if table.num_rows <= 1 or not sort_keys:
        return table
    indices = stable_sort_indices(
        table,
        sort_keys=sort_keys,
        null_placement=null_placement,
    )
    return table.take(indices)


def hash_struct_ordinal(
    table: pa.Table,
    *,
    columns: Sequence[str],
    modulus: int,
) -> pa.Array | pa.ChunkedArray:
    """Hash columns into a deterministic ordinal when kernels are available.

    Parameters
    ----------
    table
        Source table containing the columns to hash.
    columns
        Column names to hash together.
    modulus
        Modulus applied to the hash result.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Ordinal values derived from the hash.

    Raises
    ------
    RuntimeError
        If the hash kernel is unavailable.
    ValueError
        If the modulus is invalid.
    """
    if modulus <= 0:
        msg = "hash_struct_ordinal requires a positive modulus"
        raise ValueError(msg)
    if not columns:
        msg = "hash_struct_ordinal requires at least one column"
        raise ValueError(msg)
    try:
        pc.get_function("hash")
    except (AttributeError, KeyError):
        msg = "Arrow hash kernel is unavailable; upgrade pyarrow to enable it."
        raise RuntimeError(msg) from None
    struct_values = _make_struct(
        [table[column] for column in columns],
        field_names=list(columns),
    )
    hashed = require_array(call_compute("hash", [struct_values]), name="hash")
    hashed_u64 = pc.cast(hashed, pa.uint64())
    modded = require_array(
        call_compute("mod", [hashed_u64, pa.scalar(modulus, type=pa.uint64())]),
        name="mod",
    )
    return pc.cast(modded, pa.int64())


def _make_struct(
    values: Sequence[pa.Array | pa.ChunkedArray],
    *,
    field_names: Sequence[str],
) -> pa.Array | pa.ChunkedArray:
    options_factory = getattr(pc, "MakeStructOptions", None)
    options = options_factory(field_names=list(field_names)) if callable(options_factory) else None
    result = call_compute("make_struct", list(values), options=options)
    return require_array(result, name="make_struct")


__all__ = [
    "SortKey",
    "hash_struct_ordinal",
    "stable_sort_indices",
    "stable_sort_table",
]
