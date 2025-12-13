"""Shared application services for CodeIntel surfaces.

Architecture Overview
---------------------
This module provides the Service layer in the serving architecture:

::

    Transport Layer (HTTP/MCP) → Service Layer → Query Layer → Repository

Canonical Imports
-----------------
This module re-exports commonly used types for convenience:

**From bootstrap (via this module):**

    from codeintel.serving.services import BackendResource, build_backend_resource

**From conversion:**

    from codeintel.serving.services import to_domain_result
    # Or directly:
    from codeintel.serving.services.conversion import to_domain_result, to_response_result

**From transport:**

    from codeintel.serving.services import LocalTransport, HttpTransport, TransportAdapter

**From query_service:**

    from codeintel.serving.services import LocalQueryService, HttpQueryService

Key Components
~~~~~~~~~~~~~~
- ``LocalQueryService``: Wraps ``DuckDBQueryApi`` for local database access
- ``HttpQueryService``: Forwards queries to a remote HTTP API
- ``BackendResource``: Container for backend instance and configuration
- ``build_backend_resource``: Factory function for creating backend resources

Transport Adapters (v2 Pattern)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For new code, prefer using transport adapters for unified query execution:

- ``TransportAdapter``: Protocol for transport-specific execution
- ``LocalTransport``: Adapter for local DuckDB queries
- ``HttpTransport``: Adapter for HTTP API queries

Response Conversion
~~~~~~~~~~~~~~~~~~~
Use the unified conversion helpers for domain/transport model interop:

- ``to_domain_result``: Converts raw responses to domain models (service layer)
- ``to_response_result``: Converts domain models to response models (backend layer)

See ``codeintel.serving.domain_models`` for the full architecture contract.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BackendResource",
    "HttpQueryService",
    "HttpTransport",
    "LocalQueryService",
    "LocalTransport",
    "TransportAdapter",
    "build_backend_resource",
    "to_domain_result",
    "to_response_result",
]

if TYPE_CHECKING:
    BackendResource: object
    HttpQueryService: object
    HttpTransport: object
    LocalQueryService: object
    LocalTransport: object
    TransportAdapter: object
    build_backend_resource: object
    to_domain_result: object
    to_response_result: object

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BackendResource": ("codeintel.serving.bootstrap", "BackendResource"),
    "HttpQueryService": ("codeintel.serving.services.query_service", "HttpQueryService"),
    "HttpTransport": ("codeintel.serving.services.transport", "HttpTransport"),
    "LocalQueryService": ("codeintel.serving.services.query_service", "LocalQueryService"),
    "LocalTransport": ("codeintel.serving.services.transport", "LocalTransport"),
    "TransportAdapter": ("codeintel.serving.services.transport", "TransportAdapter"),
    "build_backend_resource": ("codeintel.serving.bootstrap", "build_backend_resource"),
    "to_domain_result": ("codeintel.serving.services.conversion", "to_domain_result"),
    "to_response_result": ("codeintel.serving.services.conversion", "to_response_result"),
}


def __getattr__(name: str) -> object:
    """
    Lazily import service attributes to avoid circular imports during initialization.

    Returns
    -------
    object
        Requested attribute resolved from its defining module.

    Raises
    ------
    AttributeError
        When the attribute is not registered for lazy loading.
    """
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
