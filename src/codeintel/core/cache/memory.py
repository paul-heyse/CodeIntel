"""In-memory cache implementations.

This module provides in-memory cache implementations with LRU
eviction and optional TTL support.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from codeintel.core.cache.protocol import CacheStats, CacheStatsCollector

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry[V]:
    """Entry in the cache with metadata.

    Attributes
    ----------
    value
        The cached value.
    expires_at
        Expiration timestamp, or None for no expiration.
    created_at
        When the entry was created.
    """

    value: V
    expires_at: float | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns
        -------
        bool
            True if entry is expired.
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class MemoryCache[K, V]:
    """In-memory cache with LRU eviction and TTL support.

    Provides a simple in-memory cache with configurable maximum size
    and time-to-live for entries.

    Parameters
    ----------
    max_size
        Maximum number of entries. If None, no limit.
    default_ttl_s
        Default TTL in seconds. If None, entries don't expire.

    Examples
    --------
    >>> cache: MemoryCache[str, int] = MemoryCache(max_size=100)
    >>> cache.set("key1", 42)
    >>> cache.get("key1")
    42
    >>> cache.set("key2", 100, ttl_s=60.0)  # Expires in 60 seconds
    """

    def __init__(
        self,
        *,
        max_size: int | None = None,
        default_ttl_s: float | None = None,
    ) -> None:
        """Initialize the cache.

        Parameters
        ----------
        max_size
            Maximum cache size.
        default_ttl_s
            Default TTL in seconds.
        """
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl_s = default_ttl_s
        self._stats = CacheStatsCollector(max_size=max_size)

    def get(self, key: K) -> V | None:
        """Get a cached value.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        V | None
            Cached value, or None if not found or expired.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._stats.record_miss()
            return None

        if entry.is_expired:
            self._cache.pop(key, None)
            self._stats.record_miss()
            self._stats.record_ttl_expiration()
            return None

        self._cache.move_to_end(key)
        self._stats.record_hit()
        return entry.value

    def get_or_set(self, key: K, factory: Callable[[], V]) -> V:
        """Get a cached value, or set it using a factory.

        Parameters
        ----------
        key
            Cache key.
        factory
            Callable that produces the value if not cached.

        Returns
        -------
        V
            Cached or newly computed value.
        """
        value = self.get(key)
        if value is not None:
            return value

        new_value: V = factory()
        self.set(key, new_value)
        return new_value

    def set(self, key: K, value: V, *, ttl_s: float | None = None) -> None:
        """Set a cached value.

        Parameters
        ----------
        key
            Cache key.
        value
            Value to cache.
        ttl_s
            TTL in seconds, or None to use default.
        """
        effective_ttl = ttl_s if ttl_s is not None else self._default_ttl_s
        expires_at = time.time() + effective_ttl if effective_ttl is not None else None

        if key in self._cache:
            self._cache.pop(key)

        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

        self._evict_if_needed()

    def invalidate(self, key: K) -> bool:
        """Invalidate a specific key.

        Parameters
        ----------
        key
            Key to invalidate.

        Returns
        -------
        bool
            True if key was found and removed.
        """
        entry = self._cache.pop(key, None)
        return entry is not None

    def clear(self) -> int:
        """Clear all cached values.

        Returns
        -------
        int
            Number of items cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def has(self, key: K) -> bool:
        """Check if a key is in the cache.

        Parameters
        ----------
        key
            Key to check.

        Returns
        -------
        bool
            True if key exists and is not expired.
        """
        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            self._cache.pop(key, None)
            return False
        return True

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics.

        Returns
        -------
        CacheStats
            Current cache statistics.
        """
        return self._stats.to_stats(size=len(self._cache))

    @property
    def size(self) -> int:
        """Return current cache size.

        Returns
        -------
        int
            Number of entries.
        """
        return len(self._cache)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        if self._max_size is None:
            return

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._stats.record_eviction()

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._stats.record_ttl_expiration()
        return len(expired_keys)

    def __len__(self) -> int:
        """Return number of entries.

        Returns
        -------
        int
            Cache size.
        """
        return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache.

        Parameters
        ----------
        key
            Key to check.

        Returns
        -------
        bool
            True if key exists.
        """
        return self.has(key)


__all__ = [
    "CacheEntry",
    "MemoryCache",
]
