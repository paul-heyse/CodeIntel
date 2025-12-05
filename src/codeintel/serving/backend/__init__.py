"""Transport-agnostic serving backend primitives.

Query Layer Classes
-------------------
The query layer classes provide domain-specific query implementations:

- ``FunctionQueryLayer`` - Function-related queries
- ``ProfileQueryLayer`` - Profile and module queries
- ``SubsystemQueryLayer`` - Subsystem queries
- ``DatasetQueryLayer`` - Dataset queries
"""

from __future__ import annotations

from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.dataset_backend import DatasetQueryLayer
from codeintel.serving.backend.duckdb_service import DuckDBQueryService
from codeintel.serving.backend.function_backend import FunctionQueryLayer
from codeintel.serving.backend.pagination import (
    BackendLimits,
    LimitClamp,
    OffsetClamp,
    PaginatedFetch,
    clamp_limit,
    clamp_offset,
    paginate_items,
)
from codeintel.serving.backend.profile_backend import ProfileQueryLayer
from codeintel.serving.backend.subsystem_backend import SubsystemQueryLayer

__all__ = [
    "BackendContext",
    "BackendLimits",
    "DatasetQueryLayer",
    "DuckDBQueryService",
    "DuckDBRepositories",
    "FunctionQueryLayer",
    "GraphEngineProvider",
    "LimitClamp",
    "OffsetClamp",
    "PaginatedFetch",
    "ProfileQueryLayer",
    "SubsystemQueryLayer",
    "clamp_limit",
    "clamp_offset",
    "paginate_items",
]
