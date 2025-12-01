"""Protocol surfaces for DuckDB-backed query services."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.limits import BackendLimits
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileSummaryResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
    ModuleSubsystemResponse,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    TestsForFunctionResponse,
)
from codeintel.storage.gateway import StorageGateway


class FunctionQueriesApi(Protocol):
    """Function-centric query surface."""

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> FunctionSummaryResponse:
        """Return a function summary."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphRunScope | None = None,
    ) -> HighRiskFunctionsResponse:
        """List high-risk functions."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> CallGraphNeighborsResponse:
        """Return call graph neighbors."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> TestsForFunctionResponse:
        """Return tests covering a function."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """Return an ego neighborhood."""
        ...

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """Return import edges crossing a subsystem boundary."""
        ...

    def get_function_profile(self, goid_h128: int) -> FunctionProfileResponse:
        """Return a function profile."""
        ...

    def get_function_architecture(self, goid_h128: int) -> FunctionArchitectureResponse:
        """Return function architecture metrics."""
        ...


class ProfileQueriesApi(Protocol):
    """File/module profile and summary surface."""

    def get_file_summary(
        self, *, rel_path: str, scope: GraphRunScope | None = None
    ) -> FileSummaryResponse:
        """Return a file summary."""
        ...

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        """Return a file profile."""
        ...

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        """Return a module profile."""
        ...

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        """Return module architecture metrics."""
        ...

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """Return IDE hints for a file."""
        ...


class SubsystemQueriesApi(Protocol):
    """Subsystem and hints query surface."""

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        """List subsystem summaries."""
        ...

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        """Return subsystem memberships for a module."""
        ...

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Return modules for a subsystem."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        """Search subsystems."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """Summarize a subsystem."""
        ...

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        """List subsystem profiles."""
        ...

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        """List subsystem coverage rollups."""
        ...


class DatasetQueriesApi(Protocol):
    """Dataset listing and schema surface."""

    def list_datasets(self) -> list[DatasetSpecDescriptor]:
        """List datasets."""
        ...

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """Return dataset specs."""
        ...

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Mapping[str, object]]:
        """Read dataset rows."""
        ...

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        """Return dataset schema and samples."""
        ...


class DuckDBQueryApi(Protocol):
    """Composite query protocol for LocalQueryService."""

    @property
    def gateway(self) -> StorageGateway:
        """Storage gateway backing the query service."""
        ...

    @property
    def limits(self) -> BackendLimits:
        """Backend limit configuration for clamping."""
        ...

    @property
    def functions(self) -> FunctionQueriesApi:
        """Function query helpers."""
        ...

    @property
    def modules(self) -> ProfileQueriesApi:
        """Module/file profile helpers."""
        ...

    @property
    def subsystems(self) -> SubsystemQueriesApi:
        """Subsystem query helpers."""
        ...

    @property
    def datasets(self) -> DatasetQueriesApi:
        """Dataset query helpers."""
        ...

    def __getattr__(self, name: str) -> object:
        """Allow dynamic attribute delegation for wrapped helpers."""
        ...
