"""Unified registries for observability instrumentation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypeVar, cast
from weakref import WeakKeyDictionary

from codeintel.core.singleton import SingletonHolder
from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.events import TelemetryEvent, emit_event
from codeintel.observability.instrument_cache import InstrumentCache
from codeintel.observability.semconv_keys import (
    TELEMETRY_INSTRUMENTATION_NAME,
    TELEMETRY_INSTRUMENTATION_STATUS,
)

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Meter

    from codeintel.observability.policy import ObservabilityPolicy

LOG = logging.getLogger(__name__)

InstrumentationStatus = Literal["enabled", "unavailable", "suppressed", "error"]

_T = TypeVar("_T")


@dataclass(slots=True)
class InstrumentRegistry:
    """Cache instrumentation groups by meter and name."""

    _cache: InstrumentCache[Meter, dict[str, object]] = field(default_factory=InstrumentCache)
    _lock: Lock = field(default_factory=Lock)

    def get_group(
        self,
        meter: Meter,
        group: str,
        builder: Callable[[Meter], _T],
    ) -> _T:
        """Return a cached instrument group, creating it if needed.

        Returns
        -------
        _T
            Cached or newly created instrument group.
        """
        group_map = self._cache.get_or_create(meter, dict)
        with self._lock:
            existing = group_map.get(group)
            if existing is not None:
                return cast("_T", existing)
            created = builder(meter)
            group_map[group] = created
            return created


class _InstrumentRegistryHolder(SingletonHolder[InstrumentRegistry]):
    pass


def get_instrument_registry() -> InstrumentRegistry:
    """Return the shared instrument registry.

    Returns
    -------
    InstrumentRegistry
        Process-wide instrument registry instance.
    """
    return _InstrumentRegistryHolder.get(InstrumentRegistry)


@dataclass(frozen=True, slots=True)
class InstrumentationRecord:
    """Record describing instrumentation status."""

    name: str
    status: InstrumentationStatus
    detail: str | None = None


@dataclass(slots=True)
class _Instruments:
    count: Counter


_INSTRUMENTS: WeakKeyDictionary[Meter, _Instruments] = WeakKeyDictionary()


class InstrumentationRegistry:
    """Track instrumentation enablement and errors."""

    def __init__(self) -> None:
        """Initialize the instrumentation registry."""
        self._lock = Lock()
        self._records: dict[str, InstrumentationRecord] = {}
        self._emitted_metrics: WeakKeyDictionary[Meter, bool] = WeakKeyDictionary()

    def record_enabled(self, name: str) -> None:
        """Record an enabled instrumentation entry."""
        self._record(name, status="enabled")

    def record_unavailable(self, name: str, detail: str | None = None) -> None:
        """Record an unavailable instrumentation entry."""
        self._record(name, status="unavailable", detail=detail)

    def record_suppressed(self, name: str, detail: str | None = None) -> None:
        """Record a suppressed instrumentation entry."""
        self._record(name, status="suppressed", detail=detail)

    def record_error(self, name: str, detail: str | None = None) -> None:
        """Record an instrumentation error entry."""
        self._record(name, status="error", detail=detail)

    def snapshot(self) -> tuple[InstrumentationRecord, ...]:
        """Return a stable snapshot of instrumentation records.

        Returns
        -------
        tuple[InstrumentationRecord, ...]
            Sorted tuple of instrumentation records.
        """
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda record: record.name)
        return tuple(records)

    def summary(self) -> dict[str, int]:
        """Summarize instrumentation statuses.

        Returns
        -------
        dict[str, int]
            Mapping of status name to count.
        """
        counts: dict[str, int] = {
            "enabled": 0,
            "unavailable": 0,
            "suppressed": 0,
            "error": 0,
        }
        for record in self.snapshot():
            counts[record.status] += 1
        return counts

    def emit_summary(
        self,
        *,
        policy: ObservabilityPolicy,
        logger: logging.Logger | None = None,
    ) -> None:
        """Emit a structured telemetry summary of instrumentation status."""
        payload = {
            "summary": self.summary(),
            "records": [
                {
                    "name": record.name,
                    "status": record.status,
                    "detail": record.detail,
                }
                for record in self.snapshot()
            ],
        }
        event = TelemetryEvent(
            name="telemetry.instrumentation",
            log_payload=payload,
            log_level=logging.INFO,
        )
        emit_event(
            event=event,
            span=None,
            normalizer=build_attribute_normalizer(policy),
            logger=logger or LOG,
        )

    def emit_metrics(self, meter: Meter, *, policy: ObservabilityPolicy) -> None:
        """Emit instrumentation status metrics using the supplied meter."""
        with self._lock:
            if meter in self._emitted_metrics:
                return
            self._emitted_metrics[meter] = True

        normalizer = build_attribute_normalizer(policy)
        allowed = frozenset({TELEMETRY_INSTRUMENTATION_NAME, TELEMETRY_INSTRUMENTATION_STATUS})
        instruments = _get_instruments(meter)
        for record in self.snapshot():
            attrs = normalizer.normalize(
                {
                    TELEMETRY_INSTRUMENTATION_NAME: record.name,
                    TELEMETRY_INSTRUMENTATION_STATUS: record.status,
                },
                allowed_keys=allowed,
            )
            instruments.count.add(
                1,
                attributes=attrs,
            )

    def clear(self) -> None:
        """Clear all tracked instrumentation records and emission state."""
        with self._lock:
            self._records.clear()
            self._emitted_metrics.clear()

    def _record(
        self,
        name: str,
        *,
        status: InstrumentationStatus,
        detail: str | None = None,
    ) -> None:
        record = InstrumentationRecord(name=name, status=status, detail=detail)
        with self._lock:
            self._records[name] = record


def _get_instruments(meter: Meter) -> _Instruments:
    instruments = _INSTRUMENTS.get(meter)
    if instruments is not None:
        return instruments

    instruments = _Instruments(
        count=meter.create_counter(
            "codeintel.telemetry.instrumentations",
            unit="1",
            description="Count of instrumentation statuses by name",
        )
    )
    _INSTRUMENTS[meter] = instruments
    return instruments


class _RegistryHolder(SingletonHolder[InstrumentationRegistry]):
    pass


def get_instrumentation_registry() -> InstrumentationRegistry:
    """Return the process-wide instrumentation registry.

    Returns
    -------
    InstrumentationRegistry
        Process-wide instrumentation registry instance.
    """
    return _RegistryHolder.get(InstrumentationRegistry)


__all__ = [
    "InstrumentRegistry",
    "InstrumentationRecord",
    "InstrumentationRegistry",
    "InstrumentationStatus",
    "get_instrument_registry",
    "get_instrumentation_registry",
]
