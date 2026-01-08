"""Arrow compute kernel helpers for columnar pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


def coalesce(
    values: Iterable[pa.Array | pa.ChunkedArray | pa.Scalar],
) -> pa.Array | pa.ChunkedArray:
    """Return the first non-null value across inputs.

    Parameters
    ----------
    values
        Arrays or scalars to coalesce.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Coalesced array.

    Raises
    ------
    ValueError
        If no inputs are provided.
    """
    seq = list(values)
    if not seq:
        msg = "coalesce requires at least one input"
        raise ValueError(msg)
    return require_array(call_compute("coalesce", seq), name="coalesce")


def case_when(
    cases: Sequence[tuple[pa.Array | pa.ChunkedArray, object]],
    *,
    else_: object,
) -> pa.Array | pa.ChunkedArray:
    """Return a case-when array based on boolean masks.

    Parameters
    ----------
    cases
        Sequence of (condition, value) pairs.
    else_
        Default value for rows that do not match any condition.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Resulting array.

    Raises
    ------
    ValueError
        If no cases are provided.
    """
    if not cases:
        msg = "case_when requires at least one case"
        raise ValueError(msg)
    masks = [mask for mask, _ in cases]
    values = [value for _, value in cases]
    cond_struct = _make_struct(
        masks,
        field_names=[f"cond_{idx}" for idx in range(len(masks))],
    )
    args = [cond_struct, *values, else_]
    return require_array(call_compute("case_when", args), name="case_when")


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
    "case_when",
    "coalesce",
    "hash_struct_ordinal",
    "stable_sort_indices",
]
