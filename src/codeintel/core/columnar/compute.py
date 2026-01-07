"""Arrow compute helpers for common aggregation patterns."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.core.columnar.compute_helpers import call_compute, require_array, require_scalar


def _scalar_to_int(value: pa.Scalar) -> int:
    raw = value.as_py()
    if raw is None:
        return 0
    return int(raw)


def count_true(mask: pa.Array | pa.ChunkedArray) -> int:
    """Return the number of true values in a boolean mask.

    Returns
    -------
    int
        Count of truthy entries.
    """
    result = call_compute("sum", [mask])
    scalar = require_scalar(result, name="sum")
    return _scalar_to_int(scalar)


def count_non_positive(values: pa.Array | pa.ChunkedArray) -> int:
    """Return the count of values less than or equal to zero.

    Returns
    -------
    int
        Count of non-positive values.
    """
    result = call_compute("less_equal", [values, pa.scalar(0)])
    mask = require_array(result, name="less_equal")
    return count_true(mask)


def count_distinct(values: pa.Array | pa.ChunkedArray) -> int:
    """Return the number of distinct values.

    Returns
    -------
    int
        Count of distinct values.
    """
    result = call_compute("count_distinct", [values])
    scalar = require_scalar(result, name="count_distinct")
    return _scalar_to_int(scalar)


def orphan_ref_count(
    source: pa.Array | pa.ChunkedArray,
    target: pa.Array | pa.ChunkedArray,
    *,
    allow_null: bool,
) -> int:
    """Return the number of source values not found in target.

    Returns
    -------
    int
        Count of orphan references.
    """
    in_target = call_compute(
        "is_in",
        [source],
        options=pc.SetLookupOptions(value_set=target),
    )
    present_mask = require_array(in_target, name="is_in")
    missing = require_array(call_compute("invert", [present_mask]), name="invert")
    if allow_null:
        nulls = require_array(call_compute("is_null", [source]), name="is_null")
        combined = require_array(call_compute("or_kleene", [missing, nulls]), name="or_kleene")
        return count_true(combined)
    return count_true(missing)


__all__ = [
    "count_distinct",
    "count_non_positive",
    "count_true",
    "orphan_ref_count",
]
