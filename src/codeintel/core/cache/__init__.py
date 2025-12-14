"""Unified caching infrastructure.

This module provides core caching patterns for the codebase,
including protocols, in-memory caches, and scoped caches.

Examples
--------
Using the memory cache:

>>> from codeintel.core.cache import MemoryCache
>>>
>>> cache: MemoryCache[str, int] = MemoryCache(max_size=100)
>>> cache.set("key1", 42)
>>> cache.get("key1")
42

Using snapshot-scoped cache:

>>> from codeintel.core.cache import SnapshotKey, SnapshotScopedCache
>>>
>>> cache: SnapshotScopedCache[str] = SnapshotScopedCache()
>>> key = SnapshotKey(repo="org/repo", commit="abc123")
>>> cache.set(key, "data")
>>> cache.invalidate(repo="org/repo")
1

Using cache key utilities:

>>> from codeintel.core.cache import cache_key, KeyBuilder
>>>
>>> key = cache_key("function", "repo/name", version=1)
>>> key = KeyBuilder("analytics").add("functions").add("v1").build()
"""

from codeintel.core.cache.keying import (
    CompositeKey,
    KeyBuilder,
    cache_key,
    hash_key,
)
from codeintel.core.cache.memory import (
    CacheEntry,
    MemoryCache,
)
from codeintel.core.cache.protocol import (
    CacheProtocol,
    CacheStats,
    CacheStatsCollector,
    ScopedCacheProtocol,
)
from codeintel.core.cache.scoped import (
    SnapshotKey,
    SnapshotScopedCache,
    TypedScopedCache,
)

__all__ = [
    "CacheEntry",
    "CacheProtocol",
    "CacheStats",
    "CacheStatsCollector",
    "CompositeKey",
    "KeyBuilder",
    "MemoryCache",
    "ScopedCacheProtocol",
    "SnapshotKey",
    "SnapshotScopedCache",
    "TypedScopedCache",
    "cache_key",
    "hash_key",
]
