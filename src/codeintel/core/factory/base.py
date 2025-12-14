"""Base factory implementation.

This module provides base classes for factory implementations.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from codeintel.core.factory.protocol import FactoryCreationError

log = logging.getLogger(__name__)


class BaseFactory[T]:
    """Base class for factory implementations.

    Provides common patterns for creating objects with
    logging and error handling.

    Type Parameters
    ---------------
    T
        The type of object this factory creates.

    Attributes
    ----------
    FACTORY_NAME
        Unique identifier for this factory type.

    Examples
    --------
    >>> class ConfigFactory(BaseFactory[dict[str, str]]):
    ...     FACTORY_NAME = "config"
    ...
    ...     def _do_create(self, **kwargs: object) -> dict[str, str]:
    ...         return {"env": str(kwargs.get("env", "dev"))}
    """

    FACTORY_NAME: ClassVar[str] = ""

    def create(self, **kwargs: object) -> T:
        """Create an instance.

        Parameters
        ----------
        **kwargs
            Arguments for creating the instance.

        Returns
        -------
        T
            The created instance.

        Raises
        ------
        FactoryCreationError
            If creation fails.
        """
        log.debug("Creating %s with %s", self.FACTORY_NAME, list(kwargs.keys()))

        try:
            result = self._do_create(**kwargs)
        except Exception as e:
            log.exception("Failed to create %s", self.FACTORY_NAME)
            raise FactoryCreationError(self.FACTORY_NAME, str(e)) from e
        else:
            log.debug("Created %s successfully", self.FACTORY_NAME)
            return result

    @staticmethod
    def can_create(**kwargs: object) -> bool:
        """Check if creation would succeed.

        Override this to add validation logic.

        Parameters
        ----------
        **kwargs
            Arguments to check.

        Returns
        -------
        bool
            True if creation would succeed.
        """
        _ = kwargs
        return True

    def _do_create(self, **kwargs: object) -> T:
        """Perform the actual creation.

        Subclasses must implement this method.

        Parameters
        ----------
        **kwargs
            Arguments for creating the instance.

        Raises
        ------
        NotImplementedError
            If not overridden.
        """
        msg = f"{self.__class__.__name__} must implement _do_create()"
        raise NotImplementedError(msg)


class CachingFactory[T](BaseFactory[T]):
    """Factory with instance caching.

    Caches created instances by key for reuse.

    Type Parameters
    ---------------
    T
        The type of object this factory creates.

    Examples
    --------
    >>> class ConnectionFactory(CachingFactory[Connection]):
    ...     FACTORY_NAME = "connection"
    ...
    ...     def _do_create(self, **kwargs: object) -> Connection:
    ...         return Connection(kwargs["url"])
    >>> factory = ConnectionFactory()
    >>> conn = factory.get_or_create("main", url="db://localhost")
    """

    def __init__(self) -> None:
        """Initialize the caching factory."""
        self._cache: dict[str, T] = {}

    def get_or_create(self, key: str, **kwargs: object) -> T:
        """Get a cached instance or create a new one.

        Parameters
        ----------
        key
            Cache key for the instance.
        **kwargs
            Arguments for creating the instance if not cached.

        Returns
        -------
        T
            The cached or newly created instance.
        """
        if key in self._cache:
            log.debug("Cache hit for %s: %s", self.FACTORY_NAME, key)
            return self._cache[key]

        log.debug("Cache miss for %s: %s", self.FACTORY_NAME, key)
        instance = self.create(**kwargs)
        self._cache[key] = instance
        return instance

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached instance.

        Parameters
        ----------
        key
            Cache key to invalidate.

        Returns
        -------
        bool
            True if the key was found and invalidated.
        """
        if key in self._cache:
            del self._cache[key]
            log.debug("Invalidated %s: %s", self.FACTORY_NAME, key)
            return True
        return False

    def clear(self) -> int:
        """Clear all cached instances.

        Returns
        -------
        int
            Number of instances cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        log.debug("Cleared %d cached %s instances", count, self.FACTORY_NAME)
        return count


__all__ = [
    "BaseFactory",
    "CachingFactory",
]
