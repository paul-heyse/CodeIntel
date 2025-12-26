"""Shared instrument registry for observability meters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, TypeVar, cast

from codeintel.core.singleton import SingletonHolder
from codeintel.observability.instrument_cache import InstrumentCache

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

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
        Singleton instrument registry instance.
    """
    return _InstrumentRegistryHolder.get(InstrumentRegistry)


__all__ = ["InstrumentRegistry", "get_instrument_registry"]
