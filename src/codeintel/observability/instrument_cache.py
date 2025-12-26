"""Shared instrument cache for OpenTelemetry metric instruments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar
from weakref import WeakKeyDictionary

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class InstrumentCache(Generic[K, V]):
    """Cache metric instruments keyed by a weakly-referenced object."""

    _cache: WeakKeyDictionary[K, V]
    _lock: Lock

    def __init__(self) -> None:
        self._cache = WeakKeyDictionary()
        self._lock = Lock()

    def get_or_create(self, key: K, builder: Callable[[], V]) -> V:
        """Return the cached value for a key or build and cache it.

        Returns
        -------
        V
            Cached or newly created value.
        """
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                return existing
            created = builder()
            self._cache[key] = created
            return created


__all__ = ["InstrumentCache"]
