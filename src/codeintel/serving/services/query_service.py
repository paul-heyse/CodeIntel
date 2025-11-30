"""Transport-agnostic query application services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from codeintel.serving.backend import BackendLimits, DuckDBQueryService
from codeintel.serving.backend.datasets import describe_dataset
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
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.serving.services.datasets import _HttpDatasetQueryMixin, _LocalDatasetMixin
from codeintel.serving.services.functions import (
    _FunctionQueryDelegates,
    _HttpFunctionQueryMixin,
)
from codeintel.serving.services.observability import (
    ServiceCallContext,
    ServiceCallMetrics,
    ServiceObservability,
    _observe_call,
)
from codeintel.serving.services.profiles import (
    _HttpProfileQueryMixin,
    _ProfileQueryDelegates,
)
from codeintel.serving.services.subsystems import (
    _HttpSubsystemQueryMixin,
    _SubsystemQueryDelegates,
)


class FunctionQueryApi(Protocol):
    """Function-centric query surface."""

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary for an identifier."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> CallGraphNeighborsResponse:
        """Return call graph neighbors for a function."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> TestsForFunctionResponse:
        """List tests that exercise a function."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """Return an ego neighborhood in the call graph."""
        ...

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """Return import edges crossing a subsystem boundary."""
        ...

    def get_file_summary(
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> FileSummaryResponse:
        """Return a file summary."""
        ...


class ProfileQueryApi(Protocol):
    """Profile and architecture surfaces."""

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        """Return a function profile."""
        ...

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """Return a file profile."""
        ...

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """Return a module profile."""
        ...

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        """Return architecture metrics for a function."""
        ...

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """Return architecture metrics for a module."""
        ...


class SubsystemQueryApi(Protocol):
    """Subsystem and hints surfaces."""

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """List subsystems with optional filters."""
        ...

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """Return subsystem memberships for a module."""
        ...

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """Return IDE hints for a file."""
        ...

    def get_subsystem_modules(self, *, subsystem_id: str) -> SubsystemModulesResponse:
        """Return a subsystem with member modules."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """Search subsystems."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Summarize a subsystem with optional module limit."""
        ...

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        """List subsystem profiles from docs views."""
        ...

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        """List subsystem coverage rollups from docs views."""
        ...


class DatasetQueryApi(Protocol):
    """Dataset listing and retrieval surface."""

    def list_datasets(self) -> list[DatasetDescriptor]:
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
    ) -> DatasetRowsResponse:
        """Read rows from a dataset."""
        ...

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """Return schema and samples for a dataset."""
        ...


class QueryService(
    FunctionQueryApi,
    ProfileQueryApi,
    SubsystemQueryApi,
    DatasetQueryApi,
    Protocol,
):
    """
    Composite query service consumed by HTTP, MCP, and future transports.

    All application surfaces (FastAPI, MCP, CLI) must depend on this interface
    instead of touching DuckDB or raw SQL directly.

    Implementations:
        - LocalQueryService: wraps DuckDBQueryService for local DB access.
        - HttpQueryService: forwards calls to a remote HTTP server.
    """


@dataclass
class LocalQueryService(
    _FunctionQueryDelegates,
    _ProfileQueryDelegates,
    _SubsystemQueryDelegates,
    _LocalDatasetMixin,
):
    """Application service backed by a local DuckDB query layer."""

    query: DuckDBQueryService
    dataset_tables: dict[str, str] | None = None
    describe_dataset_fn: Callable[[str, str], str] = describe_dataset
    observability: ServiceObservability | None = None
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Derive dataset registry from the query gateway when not provided."""
        if self.dataset_tables is None:
            gateway = getattr(self.query, "gateway", None)
            self.dataset_tables = dict(gateway.datasets.mapping) if gateway is not None else {}

    def _call[T](
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        """
        Invoke a query with observability tracking.

        Returns
        -------
        T
            Result returned by the wrapped callable.
        """
        self.calls.append(name)
        return _observe_call(
            self.observability,
            transport="local",
            name=name,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries,
            ),
            func=func,
        )


@dataclass
class HttpQueryService(
    _HttpFunctionQueryMixin,
    _HttpProfileQueryMixin,
    _HttpSubsystemQueryMixin,
    _HttpDatasetQueryMixin,
    QueryService,
):
    """Application service that forwards queries to a remote HTTP API."""

    request_json: Callable[[str, dict[str, object]], object]
    limits: BackendLimits
    observability: ServiceObservability | None = None


__all__ = [
    "HttpQueryService",
    "LocalQueryService",
    "QueryService",
    "ResponseMeta",
    "ServiceCallContext",
    "ServiceCallMetrics",
    "ServiceObservability",
]
