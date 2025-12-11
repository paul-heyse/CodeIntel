"""Protocol surfaces for DuckDB-backed query services.

Query Protocol Hierarchy
------------------------
This module defines backend-layer query protocols using ``GraphRunScope``
for scope parameters. These are internal protocols for the query layer.

The **canonical unified protocols** are in ``codeintel.serving.types``:

- ``FunctionQueryProtocol`` - uses ``GraphScopePayload`` (service-facing)
- ``ProfileQueryProtocol`` - uses ``GraphScopePayload`` (service-facing)
- ``SubsystemQueryProtocol`` - uses ``GraphScopePayload`` (service-facing)
- ``DatasetQueryProtocol`` - uses ``GraphScopePayload`` (service-facing)

Implementations at the backend layer accept ``GraphRunScope`` and are called
by service-layer code that converts ``GraphScopePayload`` → ``GraphRunScope``
using ``parse_graph_scope()``.

Layer Hierarchy
~~~~~~~~~~~~~~~
::

    Service Layer (GraphScopePayload)
         │
         │ parse_graph_scope()
         ▼
    Backend Layer (GraphRunScope) ← This module
         │
         ▼
    Repository Layer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.serving import domain_models as dm
    from codeintel.serving.backend.pagination import BackendLimits
    from codeintel.serving.mcp.models import DatasetSpecDescriptor
    from codeintel.storage.gateway import StorageGateway


class FunctionQueriesApi(Protocol):
    """Function-centric query surface (backend layer).

    Note: This protocol uses ``GraphRunScope`` for the scope parameter.
    Service-layer code should use ``FunctionQueryProtocol`` from
    ``codeintel.serving.types`` which uses ``GraphScopePayload``.
    """

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> dm.FunctionSummaryResult:
        """Return a function summary."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphRunScope | None = None,
    ) -> dm.HighRiskFunctionsResult:
        """List high-risk functions."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> dm.CallGraphNeighbors:
        """Return call graph neighbors."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> dm.TestsForFunctionResult:
        """Return tests covering a function."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        """Return an ego neighborhood."""
        ...

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        """Return import edges crossing a subsystem boundary."""
        ...

    def get_function_profile(self, goid_h128: int) -> dm.FunctionProfileResult:
        """Return a function profile."""
        ...

    def get_function_architecture(self, goid_h128: int) -> dm.FunctionArchitectureResult:
        """Return function architecture metrics."""
        ...


class ProfileQueriesApi(Protocol):
    """File/module profile and summary surface."""

    def get_file_summary(
        self, *, rel_path: str, scope: GraphRunScope | None = None
    ) -> dm.FileSummaryResult:
        """Return a file summary."""
        ...

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        """Return a file profile."""
        ...

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        """Return a module profile."""
        ...

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        """Return module architecture metrics."""
        ...

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        """Return IDE hints for a file."""
        ...


class SubsystemQueriesApi(Protocol):
    """Subsystem and hints query surface."""

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSummaryResult:
        """List subsystem summaries."""
        ...

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        """Return subsystem memberships for a module."""
        ...

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        """Return modules for a subsystem."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSearchResult:
        """Search subsystems."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        """Summarize a subsystem."""
        ...

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        """List subsystem profiles."""
        ...

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        """List subsystem coverage rollups."""
        ...


class DatasetQueriesApi(Protocol):
    """Dataset listing and schema surface."""

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
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

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
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
