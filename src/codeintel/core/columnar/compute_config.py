"""Shared Arrow compute options for consistent behavior."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

DEFAULT_CAST_SAFE = pc.CastOptions.safe(target_type=pa.string())
DEFAULT_SCALAR_AGG = pc.ScalarAggregateOptions(skip_nulls=True)
DEFAULT_SCALAR_AGG_ALLOW_NULL = pc.ScalarAggregateOptions(skip_nulls=False)
DEFAULT_TAKE = pc.TakeOptions(boundscheck=True)

__all__ = [
    "DEFAULT_CAST_SAFE",
    "DEFAULT_SCALAR_AGG",
    "DEFAULT_SCALAR_AGG_ALLOW_NULL",
    "DEFAULT_TAKE",
]
