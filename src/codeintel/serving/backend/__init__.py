"""Transport-agnostic serving backend primitives."""

from __future__ import annotations

from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.duckdb_service import DuckDBQueryService
from codeintel.serving.backend.pagination import (
    BackendLimits,
    ClampResult,
    LimitClamp,
    PaginatedFetch,
    clamp_limit,
    clamp_limit_value,
    clamp_offset_value,
    paginate_items,
)

__all__ = [
    "BackendContext",
    "BackendLimits",
    "ClampResult",
    "DuckDBQueryService",
    "DuckDBRepositories",
    "GraphEngineProvider",
    "LimitClamp",
    "PaginatedFetch",
    "clamp_limit",
    "clamp_limit_value",
    "clamp_offset_value",
    "paginate_items",
]
