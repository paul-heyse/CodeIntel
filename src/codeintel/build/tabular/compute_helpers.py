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


__all__ = ["array_from_compute", "call_compute", "safe_filter", "scalar_from_compute"]
