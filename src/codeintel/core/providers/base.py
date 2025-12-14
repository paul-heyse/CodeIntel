"""Base provider implementation.

This module provides base classes for resource providers with
common patterns for lazy loading and caching.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from codeintel.core.providers.protocol import (
    ProviderError,
    ProviderNotLoadedError,
)

log = logging.getLogger(__name__)


class BaseProvider[T](ABC):
    """Abstract base class for resource providers.

    Provides a standard implementation pattern for providers
    with lazy loading and caching.

    Subclasses must implement `_load()` to define the loading logic.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.

    Attributes
    ----------
    PROVIDER_NAME
        Class variable identifying this provider type.

    Examples
    --------
    >>> class ConfigProvider(BaseProvider[dict[str, str]]):
    ...     PROVIDER_NAME = "config"
    ...
    ...     def _load(self) -> dict[str, str]:
    ...         return load_config_from_file()
    """

    PROVIDER_NAME: ClassVar[str] = ""

    def __init__(self) -> None:
        """Initialize the provider."""
        self._cached: T | None = None
        self._loaded = False
        self._load_error: Exception | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if the resource has been loaded.

        Returns
        -------
        bool
            True if successfully loaded.
        """
        return self._loaded

    @property
    def provider_name(self) -> str:
        """Return the provider name.

        Returns
        -------
        str
            Provider name for identification.
        """
        return self.PROVIDER_NAME or self.__class__.__name__

    @abstractmethod
    def _load(self) -> T:
        """Load the resource.

        Subclasses implement this to define loading logic.

        Returns
        -------
        T
            The loaded resource.

        Raises
        ------
        Exception
            If loading fails.
        """
        ...

    def get(self) -> T:
        """Get the resource, loading if necessary.

        Returns
        -------
        T
            The loaded resource.

        Raises
        ------
        ProviderNotLoadedError
            If loading fails.
        """
        if self._loaded and self._cached is not None:
            return self._cached

        if self._load_error is not None:
            raise ProviderNotLoadedError(
                self.provider_name, str(self._load_error)
            ) from self._load_error

        try:
            self._cached = self._load()
        except Exception as e:
            self._load_error = e
            log.exception("Failed to load resource: %s", self.provider_name)
            raise ProviderNotLoadedError(self.provider_name, str(e)) from e
        else:
            self._loaded = True
            log.debug("Loaded resource: %s", self.provider_name)
            return self._cached

    def get_or_none(self) -> T | None:
        """Get the resource or None if unavailable.

        Returns
        -------
        T | None
            The loaded resource, or None if unavailable.
        """
        try:
            return self.get()
        except ProviderError:
            return None

    def refresh(self) -> None:
        """Force refresh of cached resource.

        Clear cached state and any error, allowing reload on next get().
        """
        self._cached = None
        self._loaded = False
        self._load_error = None
        log.debug("Refreshed provider: %s", self.provider_name)

    def invalidate(self) -> None:
        """Invalidate the cached resource.

        Alias for refresh() for compatibility with ResourceProvider protocol.
        """
        self.refresh()

    def set_preloaded(self, resource: T) -> None:
        """Set a pre-loaded resource value.

        Use this to inject a resource without triggering _load().
        Useful for testing and dependency injection.

        Parameters
        ----------
        resource
            The pre-loaded resource value.
        """
        self._cached = resource
        self._loaded = True
        self._load_error = None
        log.debug("Set preloaded resource: %s", self.provider_name)


class LazyProvider[T](BaseProvider[T]):
    """Provider that wraps a factory function.

    A simple provider that calls a factory function on first access.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.

    Examples
    --------
    >>> def load_data() -> list[str]:
    ...     return ["item1", "item2"]
    >>> provider: LazyProvider[list[str]] = LazyProvider(load_data, name="data")
    >>> provider.get()
    ['item1', 'item2']
    """

    def __init__(
        self,
        factory: Callable[[], T],
        *,
        name: str = "",
    ) -> None:
        """Initialize the lazy provider.

        Parameters
        ----------
        factory
            Callable that produces the resource.
        name
            Provider name for identification.
        """
        super().__init__()
        self._factory: Callable[[], T] = factory
        self._name = name

    @property
    def provider_name(self) -> str:
        """Return the provider name.

        Returns
        -------
        str
            Provider name.
        """
        return self._name or self.__class__.__name__

    def _load(self) -> T:
        """Load by calling the factory.

        Returns
        -------
        T
            The produced resource.
        """
        return self._factory()


class CachedProvider[T](BaseProvider[T]):
    """Provider with TTL-based caching.

    Extends BaseProvider with time-based cache expiration.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.

    Examples
    --------
    >>> class MetricsProvider(CachedProvider[dict[str, float]]):
    ...     PROVIDER_NAME = "metrics"
    ...     CACHE_TTL_S = 60.0  # Refresh every minute
    ...
    ...     def _load(self) -> dict[str, float]:
    ...         return fetch_metrics()
    """

    CACHE_TTL_S: ClassVar[float] = 0.0

    def __init__(self) -> None:
        """Initialize the cached provider."""
        super().__init__()
        self._loaded_at: float | None = None

    def get(self) -> T:
        """Get the resource, reloading if expired.

        Returns
        -------
        T
            The loaded resource.

        May raise exceptions from parent get() method.
        """
        if self._is_expired():
            self.refresh()
        return super().get()

    def _is_expired(self) -> bool:
        """Check if the cache has expired.

        Returns
        -------
        bool
            True if cache has expired.
        """
        if self._loaded_at is None or self.CACHE_TTL_S <= 0:
            return False
        return time.time() - self._loaded_at > self.CACHE_TTL_S

    def _load(self) -> T:
        """Load and record timestamp.

        Returns
        -------
        T
            The loaded resource.
        """
        result = self._do_load()
        self._loaded_at = time.time()
        return result

    @abstractmethod
    def _do_load(self) -> T:
        """Perform the actual loading.

        Subclasses implement this instead of _load().

        Returns
        -------
        T
            The loaded resource.
        """
        ...


__all__ = [
    "BaseProvider",
    "CachedProvider",
    "LazyProvider",
]
