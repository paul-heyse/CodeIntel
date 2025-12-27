"""Cache event metrics for Hamilton caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.observability.instrument_registry import get_instrument_registry
from codeintel.observability.runtime import get_observability

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter


@dataclass(slots=True)
class _CacheInstruments:
    hits: Counter
    misses: Counter
    stores: Counter
    duration_ms: Histogram


_REGISTRY = get_instrument_registry()


def _get_instruments(meter: Meter) -> _CacheInstruments:
    def _builder(inner_meter: Meter) -> _CacheInstruments:
        return _CacheInstruments(
            hits=inner_meter.create_counter(
                "codeintel.cache.hit",
                unit="1",
                description="Count of cache hit events",
            ),
            misses=inner_meter.create_counter(
                "codeintel.cache.miss",
                unit="1",
                description="Count of cache miss events",
            ),
            stores=inner_meter.create_counter(
                "codeintel.cache.store",
                unit="1",
                description="Count of cache store events",
            ),
            duration_ms=inner_meter.create_histogram(
                "codeintel.cache.duration_ms",
                unit="ms",
                description="Duration of cached node execution in milliseconds",
            ),
        )

    return _REGISTRY.get_group(meter, "cache_events", _builder)


@dataclass(frozen=True, slots=True)
class CacheEventMetrics:
    """Helper for emitting cache event metrics."""

    @staticmethod
    def record_hit(duration_ms: float | None = None) -> None:
        """Record a cache hit event.

        Parameters
        ----------
        duration_ms
            Optional duration in milliseconds for the cached node execution.
        """
        runtime = get_observability()
        if not runtime.enabled or runtime.meter is None:
            return
        instruments = _get_instruments(runtime.meter)
        instruments.hits.add(1)
        if duration_ms is not None:
            instruments.duration_ms.record(duration_ms)

    @staticmethod
    def record_miss(duration_ms: float | None = None) -> None:
        """Record a cache miss event.

        Parameters
        ----------
        duration_ms
            Optional duration in milliseconds for the cached node execution.
        """
        runtime = get_observability()
        if not runtime.enabled or runtime.meter is None:
            return
        instruments = _get_instruments(runtime.meter)
        instruments.misses.add(1)
        if duration_ms is not None:
            instruments.duration_ms.record(duration_ms)

    @staticmethod
    def record_store(duration_ms: float | None = None) -> None:
        """Record a cache store event.

        Parameters
        ----------
        duration_ms
            Optional duration in milliseconds for the cached node execution.
        """
        runtime = get_observability()
        if not runtime.enabled or runtime.meter is None:
            return
        instruments = _get_instruments(runtime.meter)
        instruments.stores.add(1)
        if duration_ms is not None:
            instruments.duration_ms.record(duration_ms)


__all__ = ["CacheEventMetrics"]
