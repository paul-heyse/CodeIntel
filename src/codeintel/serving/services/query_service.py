"""Transport-agnostic query application services.

Architecture Overview
---------------------
This module defines the **Service Layer** in the serving architecture:

::

    ┌─────────────────────────────────────────────────────────┐
    │  Transport Layer (HTTP routes, MCP backends, CLI)       │
    │  - Calls Service layer methods                          │
    │  - Converts domain models → response models             │
    └────────────────────────────────────────────────────────┘
                            ▲
                            │ domain models (dm.*)
    ┌────────────────────────────────────────────────────────┐
    │  Service Layer (this module)                            │
    │  - LocalQueryService: wraps DuckDBQueryApi              │
    │  - HttpQueryService: forwards to remote HTTP API        │
    │  - ALWAYS returns domain models (dm.*)                  │
    └────────────────────────────────────────────────────────┘
                            ▲
                            │
    ┌────────────────────────────────────────────────────────┐
    │  Query Layer (DuckDBQueryService, repositories)         │
    │  - Direct database access                               │
    │  - Graph engine integration                             │
    └────────────────────────────────────────────────────────┘

Contract
--------
All ``QueryService`` implementations MUST return domain models (``dm.*``)
from their query methods. Transport layers are responsible for converting
domain models to transport-specific response models using ``from_domain()``.

See ``codeintel.serving.domain_models`` for the full architecture contract.

Implementations
---------------
- ``LocalQueryService``: Wraps ``DuckDBQueryApi`` for local database access.
  Uses delegate mixins that call the query layer and return domain models.

- ``HttpQueryService``: Forwards queries to a remote HTTP API. Uses HTTP
  mixins that make HTTP requests, receive response models, convert them
  back to domain models via ``to_domain()``, and return domain models.

Query Protocol Hierarchy
------------------------
The **canonical unified protocols** are defined in ``codeintel.serving.types``:

- ``FunctionQueryProtocol`` - unified function query interface
- ``ProfileQueryProtocol`` - unified profile query interface
- ``SubsystemQueryProtocol`` - unified subsystem query interface
- ``DatasetQueryProtocol`` - unified dataset query interface

The protocols defined in this module (``FunctionQueryApi``, ``ProfileQueryApi``,
etc.) are **service-layer specific** and use ``GraphScopePayload`` for scope
parameters. They are compatible with the unified protocols.

For new code, prefer importing from ``codeintel.serving.types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.datasets import describe_dataset
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend.query_api import DuckDBQueryApi
    from codeintel.serving.mcp.models import (
        DatasetSpecDescriptor,
        GraphScopePayload,
    )

ResponseMeta = dm.ResponseMeta


class FunctionQueryApi(Protocol):
    """Function-centric query surface (service layer).

    Note: See ``FunctionQueryProtocol`` in ``codeintel.serving.types`` for
    the canonical unified protocol definition.
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
        """Return a function summary for an identifier."""
        ...

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        """List high-risk functions."""
        ...

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        """Return call graph neighbors for a function."""
        ...

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        """List tests that exercise a function."""
        ...

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        """Return an ego neighborhood in the call graph."""
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
        self, *, rel_path: str, scope: GraphScopePayload | None = None
    ) -> dm.FileSummaryResult:
        """Return a file summary."""
        ...


class ProfileQueryApi(Protocol):
    """Profile and architecture surfaces (service layer).

    Note: See ``ProfileQueryProtocol`` in ``codeintel.serving.types`` for
    the canonical unified protocol definition.
    """

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        """Return a function profile."""
        ...

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        """Return a file profile."""
        ...

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        """Return a module profile."""
        ...

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        """Return architecture metrics for a function."""
        ...

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        """Return architecture metrics for a module."""
        ...


class SubsystemQueryApi(Protocol):
    """Subsystem and hints surfaces (service layer).

    Note: See ``SubsystemQueryProtocol`` in ``codeintel.serving.types`` for
    the canonical unified protocol definition.
    """

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSummaryResult:
        """List subsystems with optional filters."""
        ...

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        """Return subsystem memberships for a module."""
        ...

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        """Return IDE hints for a file."""
        ...

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        """Return a subsystem with member modules and an optional limit."""
        ...

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSearchResult:
        """Search subsystems."""
        ...

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        """Summarize a subsystem with optional module limit."""
        ...

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        """List subsystem profiles from docs views."""
        ...

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        """List subsystem coverage rollups from docs views."""
        ...


class DatasetQueryApi(Protocol):
    """Dataset listing and retrieval surface (service layer).

    Note: See ``DatasetQueryProtocol`` in ``codeintel.serving.types`` for
    the canonical unified protocol definition.
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

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
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

    query: DuckDBQueryApi
    dataset_tables: dict[str, str] | None = None
    describe_dataset_fn: Callable[[str, str], str] = describe_dataset
    observability: ServiceObservability | None = None
    calls: list[str] = field(default_factory=list)
    limits: BackendLimits = field(default_factory=BackendLimits)

    def __post_init__(self) -> None:
        """Derive dataset registry from the query gateway when not provided."""
        if self.dataset_tables is None:
            try:
                gateway = self.query.gateway
            except AttributeError:
                gateway = None
            self.dataset_tables = dict(gateway.datasets.mapping) if gateway is not None else {}
        try:
            self.limits = self.query.limits
        except AttributeError:
            self.limits = BackendLimits()

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
