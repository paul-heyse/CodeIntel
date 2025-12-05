"""Transport-agnostic serving backend primitives.

Query Layer Classes
-------------------
The query layer classes were renamed from ``*Backend`` to ``*QueryLayer`` to
distinguish them from MCP backends (``DuckDBBackend``, ``HttpBackend``):

- ``FunctionQueryLayer`` (alias: ``FunctionBackend``)
- ``ProfileQueryLayer`` (alias: ``ProfileBackend``)
- ``SubsystemQueryLayer`` (alias: ``SubsystemBackend``)
- ``DatasetQueryLayer`` (alias: ``DatasetBackend``)

The old names are preserved as aliases for backward compatibility.
"""

from __future__ import annotations

from codeintel.serving.backend.core import (
    BackendContext,
    DuckDBRepositories,
    GraphEngineProvider,
)
from codeintel.serving.backend.dataset_backend import DatasetBackend, DatasetQueryLayer
from codeintel.serving.backend.duckdb_service import DuckDBQueryService
from codeintel.serving.backend.function_backend import FunctionBackend, FunctionQueryLayer
from codeintel.serving.backend.pagination import (
    BackendLimits,
    LimitClamp,
    OffsetClamp,
    PaginatedFetch,
    clamp_limit,
    clamp_offset,
    paginate_items,
)
from codeintel.serving.backend.profile_backend import ProfileBackend, ProfileQueryLayer
from codeintel.serving.backend.subsystem_backend import SubsystemBackend, SubsystemQueryLayer

__all__ = [
    "BackendContext",
    "BackendLimits",
    "DatasetBackend",
    "DatasetQueryLayer",
    "DuckDBQueryService",
    "DuckDBRepositories",
    "FunctionBackend",
    "FunctionQueryLayer",
    "GraphEngineProvider",
    "LimitClamp",
    "OffsetClamp",
    "PaginatedFetch",
    "ProfileBackend",
    "ProfileQueryLayer",
    "SubsystemBackend",
    "SubsystemQueryLayer",
    "clamp_limit",
    "clamp_offset",
    "paginate_items",
]
