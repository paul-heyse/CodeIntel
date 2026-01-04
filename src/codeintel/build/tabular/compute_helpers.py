"""Shared PyArrow compute helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc


def safe_filter(table: pa.Table, mask: pa.Array | pa.ChunkedArray) -> pa.Table:
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
    try:
        result = pc.call_function(name, list(args), options=options)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None
    if isinstance(result, pa.Scalar):
        return cast("pa.Scalar", result).as_py()
    return result


__all__ = ["safe_filter", "scalar_from_compute"]
