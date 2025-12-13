"""Serving surfaces exposing CodeIntel data via HTTP (FastAPI) and MCP protocol.

Architecture Overview
---------------------
The serving layer follows a layered architecture:

::

    ┌─────────────────────────────────────────────────────────┐
    │                   Transport Layer                        │
    │    (HTTP/FastAPI routes, MCP tools, CLI commands)       │
    └────────────────────────────┬────────────────────────────┘
                                 │
    ┌────────────────────────────▼────────────────────────────┐
    │                   Service Layer                          │
    │    QueryService (LocalQueryService, HttpQueryService)   │
    │    - Transport-agnostic business logic                  │
    │    - Observability integration                           │
    └────────────────────────────┬────────────────────────────┘
                                 │
    ┌────────────────────────────▼────────────────────────────┐
    │                    Query Layer                           │
    │    DuckDBQueryService (or graph engine queries)         │
    │    - Data access coordination                            │
    │    - Graph engine integration                            │
    └────────────────────────────┬────────────────────────────┘
                                 │
    ┌────────────────────────────▼────────────────────────────┐
    │                  Repository Layer                        │
    │    DuckDBRepositories (function, module, subsystem...)  │
    │    - Direct database access                              │
    │    - SQL execution                                       │
    └─────────────────────────────────────────────────────────┘


Key Modules
-----------
**bootstrap.py** - Service construction entry points
    - ``build_service_stack()`` - Complete service stack for servers (recommended)
    - ``build_backend_resource()`` - Backend + service bundle (recommended)
    - ``build_service_from_config()`` - Service from configuration (deprecated)

**backend/** - Query services and domain building
    - ``duckdb_service.py`` - DuckDB query service implementation
    - ``pagination.py`` - Pagination utilities and BackendLimits

**services/** - Business logic layer
    - ``query_service.py`` - QueryService protocol and implementations

**operations/** - Operation catalog and dataflow
    - ``catalog.py`` - Canonical operation definitions
    - Dataflow graph building for serving operations

**http/** - FastAPI routes and handlers

**mcp/** - MCP protocol implementation
    - ``backend.py`` - QueryBackend implementations (DuckDBBackend, HttpBackend)
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
    "HttpQueryService": ("codeintel.serving.services.query_service", "HttpQueryService"),
    "LocalQueryService": ("codeintel.serving.services.query_service", "LocalQueryService"),
    "QueryService": ("codeintel.serving.services.query_service", "QueryService"),
    "BackendResource": ("codeintel.serving.bootstrap", "BackendResource"),
    "BootstrapOptions": ("codeintel.serving.bootstrap", "BootstrapOptions"),
    "PaginatedFetch": ("codeintel.serving.backend.pagination", "PaginatedFetch"),
    "ServiceStack": ("codeintel.serving.bootstrap", "ServiceStack"),
    "build_backend_resource": ("codeintel.serving.bootstrap", "build_backend_resource"),
    "build_service_from_config": ("codeintel.serving.bootstrap", "build_service_from_config"),
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
