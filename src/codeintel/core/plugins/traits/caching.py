"""Cache awareness traits for plugins that participate in caching.

This module provides protocols and mixins for plugins that declare
cache key dependencies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheAwarePlugin(Protocol):
    """Trait for plugins that participate in caching.

    Plugins implementing this trait declare what cache keys they
    populate and consume, enabling intelligent cache management
    and dependency tracking.

    Example
    -------
    >>> class CachingPlugin(BasePlugin, CacheAwarePlugin):
    ...     @property
    ...     def cache_populates(self) -> tuple[str, ...]:
    ...         return ("function_metrics",)
    ...
    ...     @property
    ...     def cache_consumes(self) -> tuple[str, ...]:
    ...         return ("goids",)
    """

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys this plugin populates.

        Returns
        -------
        tuple[str, ...]
            Cache keys populated by this plugin.
        """
        ...

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys this plugin consumes.

        Returns
        -------
        tuple[str, ...]
            Cache keys consumed by this plugin.
        """
        ...


class CacheAwareMixin:
    """Mixin providing cache awareness to plugins.

    Use this mixin to implement CacheAwarePlugin with configurable
    cache keys via class attributes.

    Class Attributes
    ----------------
    _cache_populates
        Cache keys this plugin writes.
    _cache_consumes
        Cache keys this plugin reads.

    Example
    -------
    >>> class MyPlugin(BasePlugin, CacheAwareMixin):
    ...     _cache_populates = ("my_data",)
    ...     _cache_consumes = ("upstream_data",)
    """

    _cache_populates: tuple[str, ...] = ()
    _cache_consumes: tuple[str, ...] = ()

    @property
    def cache_populates(self) -> tuple[str, ...]:
        """Return cache keys populated by this plugin.

        Returns
        -------
        tuple[str, ...]
            Keys this plugin writes into the cache.
        """
        return self._cache_populates

    @property
    def cache_consumes(self) -> tuple[str, ...]:
        """Return cache keys consumed by this plugin.

        Returns
        -------
        tuple[str, ...]
            Keys this plugin expects to read from the cache.
        """
        return self._cache_consumes


def is_cache_aware(plugin: object) -> bool:
    """Check if a plugin implements CacheAwarePlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin participates in caching.
    """
    return isinstance(plugin, CacheAwarePlugin)


__all__ = [
    "CacheAwareMixin",
    "CacheAwarePlugin",
    "is_cache_aware",
]
