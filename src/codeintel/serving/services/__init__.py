"""Shared application services for CodeIntel surfaces.

Architecture Overview
---------------------
This module provides the Service layer in the serving architecture:

::

    Transport Layer (HTTP/MCP) → Service Layer → Query Layer → Repository

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

Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~
The ``base`` module defines abstract base classes using the template method
pattern for query delegates:

- ``BaseFunctionQueries``: Interface for function-related queries
- ``BaseProfileQueries``: Interface for profile/module queries
- ``BaseSubsystemQueries``: Interface for subsystem queries
- ``BaseDatasetQueries``: Interface for dataset queries

These serve as documentation for the intended pattern. Future refactoring
may migrate existing implementations to inherit from these bases.

See ``codeintel.serving.domain_models`` for the full architecture contract.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BackendResource",
    "BaseDatasetQueries",
    "BaseFunctionQueries",
    "BaseProfileQueries",
    "BaseSubsystemQueries",
    "HttpQueryService",
    "HttpTransport",
    "LocalQueryService",
    "LocalTransport",
    "TransportAdapter",
    "build_backend_resource",
]

if TYPE_CHECKING:
    BackendResource: object
    BaseDatasetQueries: object
    BaseFunctionQueries: object
    BaseProfileQueries: object
    BaseSubsystemQueries: object
    HttpQueryService: object
    HttpTransport: object
    LocalQueryService: object
    LocalTransport: object
    TransportAdapter: object
    build_backend_resource: object

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BackendResource": ("codeintel.serving.bootstrap", "BackendResource"),
    "BaseDatasetQueries": ("codeintel.serving.services.base", "BaseDatasetQueries"),
    "BaseFunctionQueries": ("codeintel.serving.services.base", "BaseFunctionQueries"),
    "BaseProfileQueries": ("codeintel.serving.services.base", "BaseProfileQueries"),
    "BaseSubsystemQueries": ("codeintel.serving.services.base", "BaseSubsystemQueries"),
    "HttpQueryService": ("codeintel.serving.services.query_service", "HttpQueryService"),
    "HttpTransport": ("codeintel.serving.services.transport", "HttpTransport"),
    "LocalQueryService": ("codeintel.serving.services.query_service", "LocalQueryService"),
    "LocalTransport": ("codeintel.serving.services.transport", "LocalTransport"),
    "TransportAdapter": ("codeintel.serving.services.transport", "TransportAdapter"),
    "build_backend_resource": ("codeintel.serving.bootstrap", "build_backend_resource"),
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
