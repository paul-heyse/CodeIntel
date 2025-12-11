"""HTTP transport mixin for query services."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from codeintel.serving.services.observability import (
    ServiceCallContext,
    ServiceCallMetrics,
    _observe_call,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend import BackendLimits
    from codeintel.serving.services.observability import (
        ServiceObservability,
    )

T = TypeVar("T")


class _HttpTransportMixin:
    """Shared HTTP wrapper providing observability and retry metrics."""

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


__all__ = ["_HttpTransportMixin"]
