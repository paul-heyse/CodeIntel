"""Shared types and protocols for the serving layer.

This module centralizes type definitions that are used across multiple serving
modules, helping to break circular import dependencies. All protocols and type
aliases used by both the backend and MCP layers should be defined here.

Import Pattern
--------------
Instead of importing from specific modules that might create cycles::

    from codeintel.serving.mcp.backend import QueryBackend

Use this module::

    from codeintel.serving.types import QueryBackendProtocol

Protocol Hierarchy
------------------
The protocol hierarchy follows a composable design:

**Base Protocols:**

- ``RepoCommitProtocol`` - Base for repo/commit scoped entities
- ``HasModelDump``, ``HasModelValidate``, ``HasClose`` - Pydantic utilities

**Queryable Protocols (unified service/backend interface):**

- ``FunctionQueryable`` - Function and graph operations
- ``ProfileQueryable`` - Profile and architecture operations
- ``SubsystemQueryable`` - Subsystem and hints operations
- ``DatasetQueryable`` - Dataset listing and schema operations

**Composite Protocols:**

- ``QueryServiceProtocol`` - Full service interface (all queryables)
- ``QueryBackendProtocol`` - Full backend interface (all queryables + service)

Note
----
This module consolidates the previous ``protocols.py`` module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.serving import domain_models as dm
    from codeintel.serving.mcp.models import (
        DatasetSpecDescriptor,
        GraphScopePayload,
    )
    from codeintel.serving.services.query_service import QueryService


RowDict = dict[str, object]
JsonPayload = dict[str, object] | list[object]


# =============================================================================
# Utility Protocols (Pydantic and resource management)
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


# =============================================================================
# Base Protocols
# =============================================================================


class RepoCommitProtocol(Protocol):
    """Base protocol for repo/commit scoped entities.

    This protocol provides the common interface for identifying the repository
    and commit context that all serving-layer components operate within.
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
    """Protocol for storage gateway access.

    Define the minimal interface needed by serving components
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
    """Protocol for graph engine access.

    Define the minimal interface needed for graph operations
    without requiring the full GraphEngine import.
    """

    def call_graph(self) -> object:
        """Return the call graph (NetworkX DiGraph)."""
        ...

    def import_graph(self) -> object:
        """Return the import graph (NetworkX DiGraph)."""
        ...


# =============================================================================
# Repository Protocols
# =============================================================================


class RepositoryProtocol(RepoCommitProtocol, Protocol):
    """Base protocol for all repository types."""


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


# =============================================================================
# Queryable Protocols (Unified Service/Backend Interface)
# =============================================================================


class FunctionQueryable(Protocol):
    """Unified protocol for function query operations.

    This protocol is the **single source of truth** for function-related query
    methods. Implementations include:

    - Service layer: ``LocalQueryService``, ``HttpQueryService``
    - Backend layer: ``DuckDBBackend``, ``HttpBackend``

    Note: The ``scope`` parameter uses ``GraphScopePayload | None`` at the service
    layer. Backend implementations convert this to ``GraphRunScope`` internally.
    """

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        """Return a function summary for the given identifiers."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        """List high-risk functions with optional tested-only filtering."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        """Return incoming and outgoing call graph neighbors."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        """Return tests that exercise a function."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        """Return a bounded ego neighborhood in the call graph."""
        ...

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        """Return import edges crossing a subsystem boundary."""
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> dm.FileSummaryResult:
        """Return a file summary with nested function rows."""
        ...


class ProfileQueryable(Protocol):
    """Unified protocol for profile and architecture query operations.

    This protocol defines file/module profile and architecture metrics queries.
    """

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        """Return a denormalized function profile."""
        ...

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        """Return a denormalized file profile."""
        ...

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        """Return a profile for a module."""
        ...

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        """Return architecture metrics for a function."""
        ...

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        """Return architecture metrics for a module."""
        ...

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        """Return IDE hints for a file."""
        ...


class SubsystemQueryable(Protocol):
    """Unified protocol for subsystem query operations.

    This protocol defines subsystem, hints, and search queries.
    """

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        """List inferred subsystems with optional filters."""
        ...

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        """Return subsystem memberships for a module."""
        ...

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        """Return subsystem detail and member modules."""
        ...

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        """Search subsystems by role or label."""
        ...

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        """Summarize a subsystem with optional module truncation."""
        ...

    def list_subsystem_profiles(
        self,
        *,
        limit: int | None = None,
    ) -> dm.SubsystemProfileResult:
        """List subsystem profiles from docs views."""
        ...

    def list_subsystem_coverage(
        self,
        *,
        limit: int | None = None,
    ) -> dm.SubsystemCoverageResult:
        """List subsystem coverage rollups from docs views."""
        ...


class DatasetQueryable(Protocol):
    """Unified protocol for dataset query operations.

    This protocol defines dataset listing, schema, and row retrieval queries.
    """

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        """List available datasets."""
        ...

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """Return canonical dataset contract entries."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        """Read rows from a dataset."""
        ...

    def dataset_schema(
        self,
        *,
        dataset_name: str,
        sample_limit: int = 5,
    ) -> dm.DatasetSchema:
        """Return schema and samples for a dataset."""
        ...


# =============================================================================
# Composite Protocols
# =============================================================================


class QueryServiceProtocol(
    FunctionQueryable,
    ProfileQueryable,
    SubsystemQueryable,
    DatasetQueryable,
    RepoCommitProtocol,
    Protocol,
):
    """Composite query service interface.

    This protocol combines all queryable protocols into a single interface
    for query services. Use this when type-hinting service parameters.
    """


class QueryBackendProtocol(
    FunctionQueryable,
    ProfileQueryable,
    SubsystemQueryable,
    DatasetQueryable,
    RepoCommitProtocol,
    Protocol,
):
    """Composite backend interface for MCP tools.

    This protocol combines all queryable protocols into a single interface
    for query backends. It also provides access to the underlying service.
    """

    service: QueryService


# =============================================================================
# Factory Types
# =============================================================================


ServiceFactory = Callable[..., QueryServiceProtocol]
BackendFactory = Callable[..., QueryBackendProtocol]


__all__ = [
    # Composite protocols (primary)
    "QueryBackendProtocol",
    "QueryServiceProtocol",
    # Queryable protocols
    "DatasetQueryable",
    "FunctionQueryable",
    "ProfileQueryable",
    "SubsystemQueryable",
    # Base protocols
    "RepoCommitProtocol",
    # Repository protocols
    "FunctionRepositoryProtocol",
    "ModuleRepositoryProtocol",
    "RepositoryProtocol",
    "SubsystemRepositoryProtocol",
    # Utility protocols
    "GraphEngineProtocol",
    "HasClose",
    "HasModelDump",
    "HasModelValidate",
    "ResponseMetaLike",
    "ServiceResult",
    "StorageGatewayProtocol",
    # Factory types
    "BackendFactory",
    "ServiceFactory",
    # Type aliases
    "JsonPayload",
    "RowDict",
]
