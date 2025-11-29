"""Backwards-compat shim for the DuckDB query backend.

The canonical implementation now lives in ``codeintel.serving.backend``.
This module re-exports the APIs for compatibility with existing imports.
"""

from __future__ import annotations

from codeintel.serving.backend import (
    BackendLimits,
    ClampResult,
    DuckDBQueryService,
    clamp_limit_value,
    clamp_offset_value,
)

__all__ = [
    "BackendLimits",
    "ClampResult",
    "DuckDBQueryService",
    "clamp_limit_value",
    "clamp_offset_value",
]
