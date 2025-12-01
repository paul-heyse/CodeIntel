"""Serving surfaces exposing CodeIntel data via HTTP (FastAPI) and MCP protocol.

Architecture Overview
---------------------
The serving layer is organized into these key components:

**backend/** - Core query services and response building
    - `duckdb_service.py` - DuckDB query service implementation
    - `pagination.py` - Pagination utilities and types
    - `response_builders.py` - Row-to-response transformation functions
    - `operations.py` - Operation contracts registry

**services/** - Business logic and service abstractions
    - `query_service.py` - QueryService protocol and implementations
    - `factory.py` - Service construction factories

**http/** - FastAPI routes and handlers

**mcp/** - MCP protocol implementation

**New Modules (v2 architecture):**
    - `types.py` - Shared protocols to avoid import cycles
    - `bootstrap.py` - Unified service stack construction
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BackendResource",
    "BootstrapOptions",
    "DuckDBBackend",
    "HasModelDump",
    "HttpBackend",
    "HttpQueryService",
    "LocalQueryService",
    "PaginatedFetch",
    "QueryBackend",
    "QueryService",
    "ServiceStack",
    "build_backend_resource",
    "build_service_from_config",
    "build_service_stack",
]

if TYPE_CHECKING:
    BackendResource: object
    BootstrapOptions: object
    DuckDBBackend: object
    HasModelDump: object
    HttpBackend: object
    HttpQueryService: object
    LocalQueryService: object
    PaginatedFetch: object
    QueryBackend: object
    QueryService: object
    ServiceStack: object
    build_backend_resource: object
    build_service_from_config: object
    build_service_stack: object

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DuckDBBackend": ("codeintel.serving.mcp.backend", "DuckDBBackend"),
    "HttpBackend": ("codeintel.serving.mcp.backend", "HttpBackend"),
    "QueryBackend": ("codeintel.serving.mcp.backend", "QueryBackend"),
    "HasModelDump": ("codeintel.serving.types", "HasModelDump"),
    "BackendResource": ("codeintel.serving.services.factory", "BackendResource"),
    "build_backend_resource": ("codeintel.serving.services.factory", "build_backend_resource"),
    "build_service_from_config": (
        "codeintel.serving.services.factory",
        "build_service_from_config",
    ),
    "HttpQueryService": ("codeintel.serving.services.query_service", "HttpQueryService"),
    "LocalQueryService": ("codeintel.serving.services.query_service", "LocalQueryService"),
    "QueryService": ("codeintel.serving.services.query_service", "QueryService"),
    # New v2 architecture exports
    "PaginatedFetch": ("codeintel.serving.backend.pagination", "PaginatedFetch"),
    "ServiceStack": ("codeintel.serving.bootstrap", "ServiceStack"),
    "BootstrapOptions": ("codeintel.serving.bootstrap", "BootstrapOptions"),
    "build_service_stack": ("codeintel.serving.bootstrap", "build_service_stack"),
}


def __getattr__(name: str) -> object:
    """
    Lazily import serving attributes to avoid import-time circular dependencies.

    Returns
    -------
    object
        Requested attribute loaded from its defining module.

    Raises
    ------
    AttributeError
        If the requested attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
