"""Gateway caching for storage layer.

This module provides thread-safe caching of StorageGateway instances
to enable gateway reuse across build stages without repeated
connection overhead.

The cache is keyed by the normalized StorageConfig, ensuring that
gateways with identical configurations are reused.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.storage.gateway import open_gateway

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageConfig, StorageGateway


_THREAD_GUARD_ENV_VAR = "CODEINTEL_GATEWAY_CACHE_THREAD_GUARD"


def _thread_guard_enabled() -> bool:
    value = os.environ.get(_THREAD_GUARD_ENV_VAR, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _gateway_cache_key(config: StorageConfig) -> tuple[str, str, bool, bool, bool, bool, bool]:
    """Generate a cache key for a storage configuration.

    Parameters
    ----------
    config
        Storage configuration to generate key for.

    Returns
    -------
    tuple[str, str, bool, bool, bool, bool, bool]
        Cache key tuple containing resolved paths and flags.
    """
    history = str(config.history_db_path.resolve()) if config.history_db_path is not None else ""
    return (
        str(config.db_path.resolve()),
        history,
        config.read_only,
        config.apply_schema,
        config.ensure_views,
        config.validate_schema,
        config.attach_history,
    )


@dataclass(frozen=True)
class _CachedGateway:
    gateway: StorageGateway
    thread_id: int


@dataclass
class GatewayCache:
    """Thread-safe gateway cache with lifecycle management.

    This cache stores StorageGateway instances keyed by their configuration,
    enabling reuse across pipeline stages. The cache tracks open/hit statistics
    for monitoring.

    Attributes
    ----------
    _cache
        Internal mapping from cache keys to gateways.
    _stats
        Statistics tracking opens and cache hits.
    _lock
        Thread lock for safe concurrent access.

    Examples
    --------
    >>> cache = GatewayCache()
    >>> gateway = cache.get_or_create(config)
    >>> cache.stats()
    {'opens': 1, 'hits': 0, 'size': 1}
    >>> cache.close_all()
    """

    _cache: dict[tuple[str, str, bool, bool, bool, bool, bool], _CachedGateway] = field(
        default_factory=dict
    )
    _stats: dict[str, int] = field(default_factory=lambda: {"opens": 0, "hits": 0})
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create(self, config: StorageConfig) -> StorageGateway:
        """Retrieve a cached gateway or create a new one.

        Parameters
        ----------
        config
            Storage configuration for the gateway.

        Returns
        -------
        StorageGateway
            Cached or newly created gateway.
        """
        key = _gateway_cache_key(config)
        current_thread_id = threading.get_ident()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._assert_thread_owner(cached, current_thread_id)
                self._stats["hits"] += 1
                return cached.gateway
            gateway = open_gateway(config)
            self._stats["opens"] += 1
            self._cache[key] = _CachedGateway(gateway=gateway, thread_id=current_thread_id)
            return gateway

    def close_all(self) -> None:
        """Close and clear all cached gateways.

        This method should be called at the end of pipeline execution
        to release all database connections.
        """
        with self._lock:
            for cached in self._cache.values():
                cached.gateway.close()
            self._cache.clear()
            self._stats["opens"] = 0
            self._stats["hits"] = 0

    def stats(self) -> dict[str, int]:
        """Return cache statistics for monitoring.

        Returns
        -------
        dict[str, int]
            Dictionary containing opens, hits, and current cache size.
        """
        with self._lock:
            return {
                "opens": self._stats["opens"],
                "hits": self._stats["hits"],
                "size": len(self._cache),
            }

    @staticmethod
    def _assert_thread_owner(cached: _CachedGateway, current_thread_id: int) -> None:
        if not _thread_guard_enabled():
            return
        if cached.thread_id == current_thread_id:
            return
        message = (
            "Detected cross-thread reuse of a cached StorageGateway. "
            f"Created in thread {cached.thread_id}, accessed from thread {current_thread_id}. "
            f"Disable this check by unsetting {_THREAD_GUARD_ENV_VAR}."
        )
        raise RuntimeError(message)


_cache = GatewayCache()


def get_gateway(config: StorageConfig) -> StorageGateway:
    """Retrieve a cached gateway for the given configuration.

    This is the recommended way to obtain a StorageGateway when gateway
    reuse is desired. The cache ensures that gateways with identical
    configurations are shared.

    Parameters
    ----------
    config
        Storage configuration for the gateway.

    Returns
    -------
    StorageGateway
        Cached or newly created gateway.

    Examples
    --------
    >>> from codeintel.storage.gateway import StorageConfig
    >>> config = StorageConfig.for_ingest(Path("build/db/test.duckdb"))
    >>> gateway = get_gateway(config)
    >>> gateway.config.db_path.name
    'test.duckdb'
    """
    return _cache.get_or_create(config)


def close_gateways() -> None:
    """Close and clear all cached gateways.

    Call this at the end of pipeline execution to release database connections.

    Examples
    --------
    >>> close_gateways()
    """
    _cache.close_all()


def gateway_cache_stats() -> dict[str, int]:
    """Return cache statistics for monitoring.

    Returns
    -------
    dict[str, int]
        Dictionary containing opens, hits, and current cache size.

    Examples
    --------
    >>> stats = gateway_cache_stats()
    >>> "opens" in stats and "hits" in stats and "size" in stats
    True
    """
    return _cache.stats()


def reset_gateway_cache() -> None:
    """Reset the gateway cache for testing purposes.

    This should only be called in test fixtures to ensure clean state
    between tests.
    """
    _cache.close_all()


__all__ = [
    "GatewayCache",
    "close_gateways",
    "gateway_cache_stats",
    "get_gateway",
    "reset_gateway_cache",
]
