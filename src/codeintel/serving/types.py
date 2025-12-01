"""Shared types and protocols for the serving layer.

This module centralizes type definitions that are used across multiple serving
modules, helping to break circular import dependencies. All protocols and type
aliases used by both the backend and MCP layers should be defined here.

Import Pattern
--------------
Instead of importing from specific modules that might create cycles:

    # Avoid this (can create cycles)
    from codeintel.serving.mcp.backend import QueryBackend

Use this module:

    # Prefer this
    from codeintel.serving.types import QueryBackend

Note
----
This module consolidates the previous ``protocols.py`` module. Re-exports from
``codeintel.core.types`` are provided for convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

# Re-export core types for convenience (previously from protocols.py)
from codeintel.core.types import (
    PytestCallEntry,
    PytestTestEntry,
    ScipDocument,
    ScipOccurrence,
    ScipRange,
)

# Type aliases for common patterns
RowDict = dict[str, object]
JsonPayload = dict[str, object] | list[object]


# =============================================================================
# Pydantic-related Protocols
# =============================================================================


class HasModelDump(Protocol):
    """Protocol for Pydantic models used in MCP responses."""

    def model_dump(self) -> dict[str, object]:
        """Return a dictionary representation."""
        ...


@runtime_checkable
class HasModelValidate(Protocol):
    """Protocol for types supporting Pydantic model_validate."""

    @classmethod
    def model_validate(cls, obj: object) -> HasModelValidate:
        """Validate and construct from arbitrary object."""
        ...


# =============================================================================
# Resource Management Protocols
# =============================================================================


@runtime_checkable
class HasClose(Protocol):
    """Protocol for resources with a close method."""

    def close(self) -> None:
        """Release resources."""
        ...


class ResponseMetaLike(Protocol):
    """Protocol for response metadata objects."""

    applied_limit: int | None
    truncated: bool
    messages: list[object]


class ServiceResult(Protocol):
    """Protocol for service method results with found status."""

    found: bool
    meta: ResponseMetaLike


class QueryBackendProtocol(Protocol):
    """
    Protocol for query backends (local DuckDB or remote HTTP).

    This protocol defines the common interface implemented by both
    DuckDBBackend and HttpBackend, allowing code to work with either
    without importing the concrete implementations.
    """

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Return the commit hash."""
        ...


class QueryServiceProtocol(Protocol):
    """
    Protocol for query services that provide the business logic layer.

    This protocol defines the common interface for LocalQueryService
    and HttpQueryService.
    """

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Return the commit hash."""
        ...


class StorageGatewayProtocol(Protocol):
    """
    Protocol for storage gateway access.

    Defines the minimal interface needed by serving components
    without requiring the full StorageGateway import.
    """

    @property
    def con(self) -> object:
        """Return the DuckDB connection."""
        ...

    def close(self) -> None:
        """Close the gateway and release resources."""
        ...


class GraphEngineProtocol(Protocol):
    """
    Protocol for graph engine access.

    Defines the minimal interface needed for graph operations
    without requiring the full GraphEngine import.
    """

    def call_graph(self) -> object:
        """Return the call graph (NetworkX DiGraph)."""
        ...

    def import_graph(self) -> object:
        """Return the import graph (NetworkX DiGraph)."""
        ...


class RepositoryProtocol(Protocol):
    """Base protocol for all repository types."""

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Return the commit hash."""
        ...


class FunctionRepositoryProtocol(RepositoryProtocol, Protocol):
    """Protocol for function repository operations."""

    def get_function_summary_by_goid(self, goid_h128: int) -> RowDict | None:
        """Fetch function summary by GOID."""
        ...

    def get_function_profile(self, goid_h128: int) -> RowDict | None:
        """Fetch function profile by GOID."""
        ...

    def get_function_architecture(self, goid_h128: int) -> RowDict | None:
        """Fetch function architecture by GOID."""
        ...


class ModuleRepositoryProtocol(RepositoryProtocol, Protocol):
    """Protocol for module repository operations."""

    def get_file_summary(self, rel_path: str) -> RowDict | None:
        """Fetch file summary by path."""
        ...

    def get_file_profile(self, rel_path: str) -> RowDict | None:
        """Fetch file profile by path."""
        ...

    def get_file_hints(self, rel_path: str) -> list[RowDict]:
        """Fetch IDE hints for a file."""
        ...


class SubsystemRepositoryProtocol(RepositoryProtocol, Protocol):
    """Protocol for subsystem repository operations."""

    def get_subsystem_summary(self, subsystem_id: str) -> RowDict | None:
        """Fetch subsystem summary by ID."""
        ...

    def list_subsystems(
        self, *, limit: int, role: str | None = None, query: str | None = None
    ) -> list[RowDict]:
        """List subsystems with optional filtering."""
        ...


# Type for service factory callables
ServiceFactory = Callable[..., QueryServiceProtocol]

# Type for backend factory callables
BackendFactory = Callable[..., QueryBackendProtocol]


__all__ = [
    "BackendFactory",
    "FunctionRepositoryProtocol",
    "GraphEngineProtocol",
    "HasClose",
    "HasModelDump",
    "HasModelValidate",
    "JsonPayload",
    "ModuleRepositoryProtocol",
    "PytestCallEntry",
    "PytestTestEntry",
    "QueryBackendProtocol",
    "QueryServiceProtocol",
    "RepositoryProtocol",
    "ResponseMetaLike",
    "RowDict",
    "ScipDocument",
    "ScipOccurrence",
    "ScipRange",
    "ServiceFactory",
    "ServiceResult",
    "StorageGatewayProtocol",
    "SubsystemRepositoryProtocol",
]
