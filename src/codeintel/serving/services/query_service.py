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
The canonical unified protocols are defined in ``codeintel.serving.types``:

- ``FunctionQueryable`` - unified function query interface
- ``ProfileQueryable`` - unified profile query interface
- ``SubsystemQueryable`` - unified subsystem query interface
- ``DatasetQueryable`` - unified dataset query interface

The ``QueryService`` composite protocol in this module combines all queryables.
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
from codeintel.serving.types import (
    DatasetQueryable,
    FunctionQueryable,
    ProfileQueryable,
    SubsystemQueryable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend.query_api import DuckDBQueryApi

ResponseMeta = dm.ResponseMeta


class QueryService(
    FunctionQueryable,
    ProfileQueryable,
    SubsystemQueryable,
    DatasetQueryable,
    Protocol,
):
    """Composite query service consumed by HTTP, MCP, and future transports.

    All application surfaces (FastAPI, MCP, CLI) must depend on this interface
    instead of touching DuckDB or raw SQL directly.

    This protocol combines all queryable protocols from ``codeintel.serving.types``:

    - ``FunctionQueryable`` - function and graph queries
    - ``ProfileQueryable`` - profile and architecture queries
    - ``SubsystemQueryable`` - subsystem and hints queries
    - ``DatasetQueryable`` - dataset listing and row queries

    Implementations
    ---------------
    - ``LocalQueryService``: wraps DuckDBQueryService for local DB access.
    - ``HttpQueryService``: forwards calls to a remote HTTP server.
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
