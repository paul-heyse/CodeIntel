"""Shared PyArrow compute helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from pyarrow.compute import Expression as ComputeExpression
else:
    ComputeExpression = object


def safe_filter(
    table: pa.Table,
    mask: pa.Array | pa.ChunkedArray | ComputeExpression,
) -> pa.Table:
    """Return a filtered table, falling back to the input on Arrow errors.

    Returns
    -------
    pa.Table
        Filtered table or the original on failure.
    """
    try:
        return table.filter(mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return table


def call_compute(
    name: str,
    args: Sequence[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    """Invoke an Arrow compute kernel and return the raw result.

    Returns
    -------
    object | None
        Compute kernel result or None on failure.
    """
    try:
        return pc.call_function(name, list(args), options=options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None


def array_from_compute(
    name: str,
    args: Sequence[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> pa.Array | pa.ChunkedArray | None:
    """Compute a kernel result and return it when it is array-like.

    Returns
    -------
    pa.Array | pa.ChunkedArray | None
        Array result or None on failure/unsupported kernels.
    """
    result = call_compute(name, args, options=options)
    if isinstance(result, (pa.Array, pa.ChunkedArray)):
        return result
    return None


def scalar_from_compute(
    name: str,
    args: Sequence[object],
    *,
    options: pc.FunctionOptions | None = None,
) -> object | None:
    """Compute a scalar result and return its Python value when available.

    Returns
    -------
    object | None
        Python scalar from the compute kernel, or None on failure.
    """
    result = call_compute(name, args, options=options)
    if result is None:
        return None
    if isinstance(result, pa.Scalar):
        return cast("pa.Scalar", result).as_py()
    return result


def cast_options(
    target_type: pa.DataType,
    *,
    safe: bool,
) -> pc.CastOptions:
    """Return Arrow cast options with explicit safety configuration.

    Returns
    -------
    pyarrow.compute.CastOptions
        Cast options configured for safe or unsafe conversions.
    """
    if safe:
        return pc.CastOptions(
            target_type=target_type,
            allow_int_overflow=False,
            allow_time_truncate=False,
            allow_time_overflow=False,
            allow_decimal_truncate=False,
            allow_float_truncate=False,
            allow_invalid_utf8=False,
        )
    return pc.CastOptions(
        target_type=target_type,
        allow_int_overflow=True,
        allow_time_truncate=True,
        allow_time_overflow=True,
        allow_decimal_truncate=True,
        allow_float_truncate=True,
        allow_invalid_utf8=True,
    )


def cast_array(
    values: pa.Array | pa.ChunkedArray,
    target_type: pa.DataType,
    *,
    safe: bool = False,
) -> pa.Array | pa.ChunkedArray:
    """Cast an Arrow array with explicit cast options.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Casted array.
    """
    options = cast_options(target_type, safe=safe)
    return pc.cast(values, options=options)


def sort_options(
    sort_keys: Sequence[tuple[str, str]],
    *,
    null_placement: str = "at_end",
) -> pc.SortOptions:
    """Return Arrow sort options for compute kernels.

    Returns
    -------
    pyarrow.compute.SortOptions
        Sort options configured with null placement.
    """
    try:
        return pc.SortOptions(sort_keys=sort_keys, null_placement=null_placement)
    except TypeError:
        return pc.SortOptions(sort_keys=sort_keys)


def take_options(*, boundscheck: bool = True) -> pc.TakeOptions:
    """Return Arrow take options for compute kernels.

    Returns
    -------
    pyarrow.compute.TakeOptions
        Take options configured with bounds checking.
    """
    return pc.TakeOptions(boundscheck=boundscheck)


def take_array(
    values: pa.Array | pa.ChunkedArray,
    indices: pa.Array | pa.ChunkedArray,
    *,
    boundscheck: bool = True,
) -> pa.Array | pa.ChunkedArray:
    """Take values at indices using configured bounds checking.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Result array after applying the take operation.
    """
    options = take_options(boundscheck=boundscheck)
    return pc.call_function("take", [values, indices], options=options)


__all__ = [
    "array_from_compute",
    "call_compute",
    "cast_array",
    "cast_options",
    "safe_filter",
    "scalar_from_compute",
    "sort_options",
    "take_array",
    "take_options",
]
