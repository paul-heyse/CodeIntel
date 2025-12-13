"""Transport adapters for unified local and HTTP query execution.

Architecture Overview
---------------------
This module provides a transport abstraction layer that unifies local (DuckDB)
and remote (HTTP) query execution patterns. Both transport types implement
the same ``TransportAdapter`` protocol, allowing domain service classes to
work with either transport without code duplication.

::

    ┌─────────────────────────────────────────────────────────┐
    │  Domain Services (FunctionQueryService, etc.)           │
    │  - Uses TransportAdapter for all query execution        │
    │  - Contains domain-specific business logic              │
    └─────────────────────────────────────────────────────────┘
                            │
                            │ call()
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Transport Adapters                                     │
    │  - LocalTransport: wraps DuckDBQueryApi                 │
    │  - HttpTransport: wraps HTTP request_json callable      │
    │  - _HttpTransportMixin: mixin for HTTP query services   │
    │  - Handles observability and error wrapping             │
    └─────────────────────────────────────────────────────────┘

See ``codeintel.serving.domain_models`` for the full architecture contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

from codeintel.serving.backend import BackendLimits
from codeintel.serving.services.observability import (
    ServiceCallContext,
    ServiceCallMetrics,
    _observe_call,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend.query_api import DuckDBQueryApi
    from codeintel.serving.services.observability import (
        ServiceObservability,
    )

T = TypeVar("T")


class TransportAdapter(Protocol):
    """Protocol for transport-specific query execution.

    Transport adapters wrap the underlying query mechanism (DuckDB or HTTP)
    and provide consistent observability and error handling.

    Implementations
    ---------------
    - ``LocalTransport``: Wraps ``DuckDBQueryApi`` for local database queries
    - ``HttpTransport``: Wraps an HTTP request callable for remote queries
    """

    def call[T](
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        """Execute a query operation through the transport.

        Parameters
        ----------
        operation
            Name of the operation for observability tracking.
        executor
            Callable that performs the actual query and returns the result.
        dataset
            Optional dataset name for context.
        schema_version
            Optional schema version for context.
        retries
            Optional retry count for context.

        Returns
        -------
        T
            Result returned by the executor.

        Raises
        ------
        ProblemError
            When the operation surfaces a domain problem.
        RuntimeError
            When runtime failures occur.
        ValueError
            When invalid inputs are provided.
        OSError
            When I/O issues occur.
        TimeoutError
            When the operation exceeds a timeout.
        """
        ...


@dataclass
class LocalTransport:
    """Transport adapter for local DuckDB queries.

    This adapter wraps a ``DuckDBQueryApi`` instance and provides consistent
    observability tracking for all local query operations.

    Parameters
    ----------
    query
        The DuckDB query API to use for operations.
    observability
        Optional observability configuration for metrics/logging.
    limits
        Backend limits configuration.
    """

    query: DuckDBQueryApi
    observability: ServiceObservability | None = None
    limits: BackendLimits = field(default_factory=BackendLimits)

    def call[T](
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        """Execute a local DuckDB query with observability.

        Parameters
        ----------
        operation
            Name of the operation for observability tracking.
        executor
            Callable that performs the query using self.query.
        dataset
            Optional dataset name for context.
        schema_version
            Optional schema version for context.
        retries
            Optional retry count for context.

        Returns
        -------
        T
            Result returned by the executor.
        """
        return _observe_call(
            self.observability,
            transport="local",
            name=operation,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries,
            ),
            func=executor,
        )


@dataclass
class HttpTransport:
    """Transport adapter for HTTP API queries.

    This adapter wraps an HTTP request callable and provides consistent
    observability tracking for all remote query operations.

    Parameters
    ----------
    request_json
        Callable that makes HTTP requests and returns JSON responses.
        Signature: (path: str, params: dict) -> object
    observability
        Optional observability configuration for metrics/logging.
    limits
        Backend limits configuration.
    """

    request_json: Callable[[str, dict[str, object]], object]
    observability: ServiceObservability | None = None
    limits: BackendLimits = field(default_factory=BackendLimits)

    def call[T](
        self,
        operation: str,
        executor: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
        retries: int | None = None,
    ) -> T:
        """Execute an HTTP query with observability.

        Parameters
        ----------
        operation
            Name of the operation for observability tracking.
        executor
            Callable that performs the HTTP request using self.request_json.
        dataset
            Optional dataset name for context.
        schema_version
            Optional schema version for context.
        retries
            Optional retry count for context.

        Returns
        -------
        T
            Result returned by the executor.
        """
        return _observe_call(
            self.observability,
            transport="http",
            name=operation,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries,
            ),
            func=executor,
        )


class _HttpTransportMixin:
    """Shared HTTP wrapper providing observability and retry metrics.

    This mixin is used by HTTP query service implementations to provide
    consistent observability tracking for HTTP requests. It expects the
    class to have ``request_json``, ``limits``, and ``observability``
    attributes.

    Attributes
    ----------
    request_json
        Callable for making HTTP requests.
    limits
        Backend limits configuration.
    observability
        Optional observability configuration.
    """

    request_json: Callable[[str, dict[str, object]], object]
    limits: BackendLimits
    observability: ServiceObservability | None

    def _http_call(
        self,
        name: str,
        func: Callable[[], T],
        *,
        dataset: str | None = None,
        schema_version: str | None = None,
    ) -> T:
        """
        Invoke an HTTP call with observability tracking.

        Parameters
        ----------
        name
            Operation name for logging.
        func
            Callable that performs the HTTP request.
        dataset
            Dataset name when applicable.
        schema_version
            Schema version used for the request.

        Returns
        -------
        T
            Parsed HTTP response payload.
        """
        backend = getattr(self.request_json, "__self__", None)
        retries = getattr(backend, "last_retry_attempts", None)
        result = _observe_call(
            self.observability,
            transport="http",
            name=name,
            context=ServiceCallContext(
                dataset=dataset,
                schema_version=schema_version,
                retries=retries if isinstance(retries, int) else None,
            ),
            func=func,
        )
        if retries and self.observability is not None:
            self.observability.record(
                ServiceCallMetrics(
                    name=f"{name}_retries",
                    transport="http",
                    duration_ms=0.0,
                    dataset=dataset,
                    retries=retries,
                    schema_version=schema_version,
                )
            )
        return result


__all__ = [
    "HttpTransport",
    "LocalTransport",
    "TransportAdapter",
    "_HttpTransportMixin",
]
