"""Runtime bundle registry with LRU caching."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock

from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle, RuntimeKey


@dataclass
class RuntimeRegistry:
    """In-process cache for runtime bundles keyed by RuntimeKey."""

    max_entries: int = 4
    _lock: Lock = field(default_factory=Lock, init=False)
    _entries: OrderedDict[RuntimeKey, HamiltonRuntimeBundle] = field(
        default_factory=OrderedDict, init=False
    )

    def get_or_create(
        self,
        key: RuntimeKey,
        factory: Callable[[], HamiltonRuntimeBundle],
    ) -> HamiltonRuntimeBundle:
        """Return cached runtime bundle or create and store it.

        Returns
        -------
        HamiltonRuntimeBundle
            Cached or newly created runtime bundle.
        """
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing

        runtime = factory()

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            self._entries[key] = runtime
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return runtime


__all__ = ["RuntimeRegistry"]
