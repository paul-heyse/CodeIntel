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
    LimitClamp,
    OffsetClamp,
    PaginatedFetch,
    clamp_limit,
    clamp_offset,
    paginate_items,
)

__all__ = [
    "BackendContext",
    "BackendLimits",
    "DuckDBQueryService",
    "DuckDBRepositories",
    "GraphEngineProvider",
    "LimitClamp",
    "OffsetClamp",
    "PaginatedFetch",
    "clamp_limit",
    "clamp_offset",
    "paginate_items",
]
