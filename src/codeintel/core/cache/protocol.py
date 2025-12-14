"""Cache protocol definitions.

This module defines the core protocols for cache implementations,
providing a standardized interface for caching operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CacheStats:
    """Statistics for cache performance monitoring.

    Attributes
    ----------
    hits
        Number of cache hits.
    misses
        Number of cache misses.
    size
        Current number of items in cache.
    evictions
        Number of items evicted.

    Examples
    --------
    >>> stats = CacheStats(hits=100, misses=20, size=50)
    >>> stats.hit_rate
    0.8333333333333334
    """

    hits: int = 0
    misses: int = 0
    size: int = 0
    evictions: int = 0
    max_size: int | None = None
    ttl_expirations: int = 0

    @property
    def total_requests(self) -> int:
        """Return total number of cache requests.

        Returns
        -------
        int
            Total hits + misses.
        """
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns
        -------
        float
            Hit rate as a ratio (0.0 to 1.0).
        """
        total = self.total_requests
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate.

        Returns
        -------
        float
            Miss rate as a ratio (0.0 to 1.0).
        """
        return 1.0 - self.hit_rate


@runtime_checkable
class CacheProtocol[K, V](Protocol):
    """Protocol for cache implementations.

    Caches provide a consistent interface for storing and retrieving
    values with optional time-to-live support.

    Type Parameters
    ---------------
    K
        Key type.
    V
        Value type.

    Examples
    --------
    >>> class SimpleCache:
    ...     def get(self, key: str) -> str | None:
    ...         return self._store.get(key)
    ...
    ...     def set(self, key: str, value: str, *, ttl_s: float | None = None) -> None:
    ...         self._store[key] = value
    ...
    ...     def invalidate(self, key: str) -> bool:
    ...         return self._store.pop(key, None) is not None
    ...
    ...     def clear(self) -> int:
    ...         count = len(self._store)
    ...         self._store.clear()
    ...         return count
    ...
    ...     @property
    ...     def stats(self) -> CacheStats:
    ...         return CacheStats(size=len(self._store))
    """

    def get(self, key: K) -> V | None:
        """Get a cached value.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        V | None
            Cached value, or None if not found.
        """
        ...

    def set(self, key: K, value: V, *, ttl_s: float | None = None) -> None:
        """Set a cached value.

        Parameters
        ----------
        key
            Cache key.
        value
            Value to cache.
        ttl_s
            Optional time-to-live in seconds.
        """
        ...

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
        ...

    def clear(self) -> int:
        """Clear all cached values.

        Returns
        -------
        int
            Number of items cleared.
        """
        ...

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics.

        Returns
        -------
        CacheStats
            Current cache statistics.
        """
        ...


@runtime_checkable
class ScopedCacheProtocol[K, V](Protocol):
    """Protocol for scoped caches with filtered invalidation.

    Extends basic caching with the ability to invalidate based on
    scope criteria (e.g., by repository, by timestamp).

    Type Parameters
    ---------------
    K
        Key type (typically a composite key like SnapshotKey).
    V
        Value type.
    """

    def get(self, key: K) -> V | None:
        """Get a cached value.

        Parameters
        ----------
        key
            Cache key.

        Returns
        -------
        V | None
            Cached value, or None if not found.
        """
        ...

    def set(self, key: K, value: V) -> None:
        """Set a cached value.

        Parameters
        ----------
        key
            Cache key.
        value
            Value to cache.
        """
        ...

    def invalidate_matching(self, **criteria: object) -> int:
        """Invalidate entries matching the given criteria.

        Parameters
        ----------
        **criteria
            Key-value pairs to match against cached keys.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        ...

    def clear(self) -> int:
        """Clear all cached values.

        Returns
        -------
        int
            Number of items cleared.
        """
        ...


@dataclass
class CacheStatsCollector:
    """Mutable collector for cache statistics.

    Use this in cache implementations to track hits, misses, etc.

    Examples
    --------
    >>> collector = CacheStatsCollector()
    >>> collector.record_hit()
    >>> collector.record_miss()
    >>> collector.to_stats(size=10)
    CacheStats(hits=1, misses=1, size=10, evictions=0, max_size=None, ttl_expirations=0)
    """

    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)
    _evictions: int = field(default=0, repr=False)
    _ttl_expirations: int = field(default=0, repr=False)
    max_size: int | None = None

    def record_hit(self) -> None:
        """Record a cache hit."""
        self._hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self._evictions += 1

    def record_ttl_expiration(self) -> None:
        """Record a TTL expiration."""
        self._ttl_expirations += 1

    def to_stats(self, *, size: int) -> CacheStats:
        """Convert to immutable CacheStats.

        Parameters
        ----------
        size
            Current cache size.

        Returns
        -------
        CacheStats
            Immutable statistics snapshot.
        """
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=size,
            evictions=self._evictions,
            max_size=self.max_size,
            ttl_expirations=self._ttl_expirations,
        )

    def reset(self) -> None:
        """Reset all statistics."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._ttl_expirations = 0


__all__ = [
    "CacheProtocol",
    "CacheStats",
    "CacheStatsCollector",
    "ScopedCacheProtocol",
]
