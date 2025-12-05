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
    from codeintel.serving.types import QueryBackendProtocol

Protocol Hierarchy
------------------
Backend protocols define the interface for query backends:

- ``BaseBackendProtocol`` - Base protocol with service accessor
- ``FunctionBackendProtocol`` - Function and graph operations
- ``ProfileBackendProtocol`` - Profile and architecture operations
- ``SubsystemBackendProtocol`` - Subsystem and hints operations
- ``DatasetBackendProtocol`` - Dataset listing and schema operations

Note
----
This module consolidates the previous ``protocols.py`` module. Re-exports from
``codeintel.core.types`` are provided for convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.serving import domain_models as dm
    from codeintel.serving.mcp.models import (
        CallGraphNeighborsResponse,
        DatasetDescriptor,
        DatasetRowsResponse,
        DatasetSchemaResponse,
        DatasetSpecDescriptor,
        FileHintsResponse,
        FileProfileResponse,
        FileSummaryResponse,
        FunctionArchitectureResponse,
        FunctionProfileResponse,
        FunctionSummaryResponse,
        GraphNeighborhoodResponse,
        GraphScopePayload,
        HighRiskFunctionsResponse,
        ImportBoundaryResponse,
        ModuleArchitectureResponse,
        ModuleProfileResponse,
        ModuleSubsystemResponse,
        SubsystemModulesResponse,
        SubsystemSearchResponse,
        SubsystemSummaryResponse,
        TestsForFunctionResponse,
    )
    from codeintel.serving.services.query_service import QueryService

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


# =============================================================================
# Unified Query Protocols (single source of truth)
# =============================================================================
# These protocols unify the query interfaces across the serving layer.
# - Service layer: LocalQueryService, HttpQueryService use these
# - Backend layer: DuckDBQueryService implementations conform to these
#
# Backward-compatible aliases are maintained in:
# - backend/query_api.py (FunctionQueriesApi, etc.)
# - services/query_service.py (FunctionQueryApi, etc.)


class FunctionQueryProtocol(Protocol):
    """Unified protocol for function query operations.

    This protocol is the **single source of truth** for function-related query
    methods. Implementations include:

    - Service layer: ``LocalQueryService``, ``HttpQueryService``
    - Backend layer: ``DuckDBQueryService`` (via ``FunctionQueryLayer``)

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


class ProfileQueryProtocol(Protocol):
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


class SubsystemQueryProtocol(Protocol):
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


class DatasetQueryProtocol(Protocol):
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
# Backend Protocols (moved from mcp/backend.py for centralization)
# =============================================================================


class BaseBackendProtocol(Protocol):
    """Base backend interface providing shared service access."""

    service: QueryService


class FunctionBackendProtocol(BaseBackendProtocol, Protocol):
    """Function and graph operations surfaced by backends."""

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: object | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary from analytics and docs views."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: object | None = None,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions with optional tested-only filtering."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: object | None = None,
    ) -> CallGraphNeighborsResponse:
        """Return incoming and outgoing call graph neighbors."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """Return a bounded ego neighborhood in the call graph."""
        ...

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """Return import graph edges crossing a subsystem boundary."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: object | None = None,
    ) -> TestsForFunctionResponse:
        """List tests that exercised a function."""
        ...

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: object | None = None,
    ) -> FileSummaryResponse:
        """Return a file summary with nested function rows."""
        ...


class ProfileBackendProtocol(BaseBackendProtocol, Protocol):
    """Profile and architecture operations surfaced by backends."""

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """Return a denormalized function profile."""
        ...

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """Return a denormalized file profile."""
        ...

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """Return a profile for a module."""
        ...

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """Return architecture metrics for a function."""
        ...

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """Return architecture metrics for a module."""
        ...


class SubsystemBackendProtocol(BaseBackendProtocol, Protocol):
    """Subsystem, IDE hints, and search operations."""

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """List inferred subsystems with optional filters."""
        ...

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """Return subsystem memberships for a module."""
        ...

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """Return IDE-focused hints for a file."""
        ...

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Return subsystem detail and member modules."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """Search subsystems by role or label."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Summarize a subsystem with optional module truncation."""
        ...


class DatasetBackendProtocol(BaseBackendProtocol, Protocol):
    """Dataset listing and schema operations surfaced by backends."""

    def list_datasets(self) -> list[DatasetDescriptor]:
        """List datasets available to browse."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatasetRowsResponse:
        """Read a slice of rows from a dataset."""
        ...

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """Return canonical dataset specifications."""
        ...

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """Return schema and sample rows for a dataset."""
        ...


class AggregatedBackendProtocol(
    DatasetBackendProtocol,
    FunctionBackendProtocol,
    ProfileBackendProtocol,
    SubsystemBackendProtocol,
    Protocol,
):
    """Aggregated backend interface consumed by MCP tools.

    This protocol combines all domain-specific backend protocols into a single
    interface. Use this when a component needs access to all backend operations.
    """


__all__ = [
    "AggregatedBackendProtocol",
    "BackendFactory",
    "BaseBackendProtocol",
    "DatasetBackendProtocol",
    "DatasetQueryProtocol",
    "FunctionBackendProtocol",
    "FunctionQueryProtocol",
    "FunctionRepositoryProtocol",
    "GraphEngineProtocol",
    "HasClose",
    "HasModelDump",
    "HasModelValidate",
    "JsonPayload",
    "ModuleRepositoryProtocol",
    "ProfileBackendProtocol",
    "ProfileQueryProtocol",
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
    "SubsystemBackendProtocol",
    "SubsystemQueryProtocol",
    "SubsystemRepositoryProtocol",
]
