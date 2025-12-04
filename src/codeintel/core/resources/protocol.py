"""Unified resource provider protocol.

This module defines the core protocol for resource providers that work
with both graph and analytics plugins. It includes:

- ResourceProvider: Protocol for lazy resource loading
- ResourceProviderBase: Simple base class with caching
- LazyResource: Extended base class with error tracking and optional access
- ResourceError: Base exception for resource-related errors
- ResourceNotLoadedError: Exception for lazy resource load failures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


# =============================================================================
# Exceptions
# =============================================================================


class ResourceError(Exception):
    """Base exception for resource-related errors.

    All resource-specific exceptions should inherit from this class
    to enable unified exception handling across subsystems.
    """


class ResourceNotLoadedError(ResourceError):
    """Resource has not been loaded yet.

    This exception is raised when attempting to access a lazy resource
    that failed to load or has not been loaded yet.

    Attributes
    ----------
    resource_type
        Name of the resource type that failed to load.
    reason
        Optional reason describing why the resource is not loaded.
    """

    def __init__(self, resource_type: str, reason: str | None = None) -> None:
        """Initialize the error.

        Parameters
        ----------
        resource_type
            Name of the resource type.
        reason
            Optional reason the resource is not loaded.
        """
        message = f"Resource not loaded: {resource_type}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)
        self.resource_type = resource_type
        self.reason = reason


# =============================================================================
# Protocols and Base Classes
# =============================================================================


@runtime_checkable
class ResourceProvider(Protocol[T_co]):
    """Unified protocol for resource providers.

    Resource providers give plugins access to shared data (graphs, catalogs,
    ASTs, etc.) through a consistent interface. This protocol is used by
    both graph and analytics plugins.

    The provider protocol uses lazy loading - the `get()` method loads and
    returns the resource on demand.

    Type Parameters
    ---------------
    T_co
        The type of resource this provider produces (covariant).

    Attributes
    ----------
    RESOURCE_NAME
        Class variable identifying this resource type for lookup.
    """

    RESOURCE_NAME: ClassVar[str]

    def get(self) -> T_co:
        """Load and return the resource.

        This method may perform lazy loading on first call. Subsequent
        calls should return the cached resource.

        Returns
        -------
        T_co
            The loaded resource.
        """
        ...

    def invalidate(self) -> None:
        """Invalidate the cached resource.

        Providers that support caching should implement this to force
        a reload on the next `get()` call. Providers without caching
        can implement this as a no-op.
        """
        ...


class ResourceProviderBase[T]:
    """Base class for resource providers with common functionality.

    This base class provides a standard implementation pattern for
    resource providers with lazy loading and caching.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.
    """

    RESOURCE_NAME: ClassVar[str] = ""

    def __init__(self) -> None:
        """Initialize the provider."""
        self._cached: T | None = None

    def get(self) -> T:
        """Load and return the resource, caching the result.

        Returns
        -------
        T
            The loaded resource.
        """
        if self._cached is None:
            self._cached = self._load()
        return self._cached

    def _load(self) -> T:
        """Load the resource.

        Subclasses must implement this method to provide the actual
        loading logic.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        msg = f"{self.__class__.__name__} must implement _load()"
        raise NotImplementedError(msg)

    def invalidate(self) -> None:
        """Invalidate the cached resource.

        Call this to force a reload on the next `get()` call.
        """
        self._cached = None


class LazyResource[T](ABC):
    """Abstract base class for lazy resource providers with error tracking.

    Provides a standard implementation of lazy loading with caching and
    error tracking. Subclasses implement `_load()` to define how the
    resource is loaded.

    This class provides additional features beyond ResourceProviderBase:

    - ``is_loaded`` property to check load status
    - ``get_or_none()`` for optional access without exceptions
    - ``set_preloaded()`` for dependency injection in tests
    - Error tracking for repeated failures

    Type Parameters
    ---------------
    T
        The type of resource this provider manages.

    Attributes
    ----------
    RESOURCE_NAME
        ClassVar identifying this resource type for registry lookup.
        Subclasses should override this with their specific name.

    Example
    -------
    >>> class ConfigProvider(LazyResource[dict[str, str]]):
    ...     RESOURCE_NAME = "config"
    ...
    ...     def _load(self) -> dict[str, str]:
    ...         return {"key": "value"}
    >>> provider = ConfigProvider("app_config")
    >>> config = provider.get()  # Loads on first access
    >>> provider.is_loaded
    True
    """

    RESOURCE_NAME: ClassVar[str] = ""

    def __init__(self, name: str) -> None:
        """Initialize the lazy resource.

        Parameters
        ----------
        name
            Human-readable name for the resource, used in error messages.
        """
        self._name = name
        self._resource: T | None = None
        self._loaded = False
        self._load_error: Exception | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if the resource has been loaded.

        Returns
        -------
        bool
            True if the resource has been successfully loaded.
        """
        return self._loaded

    @property
    def resource_name(self) -> str:
        """Return the resource name, preferring RESOURCE_NAME ClassVar.

        Returns
        -------
        str
            The resource name for identification.
        """
        if self.RESOURCE_NAME:
            return self.RESOURCE_NAME
        return self._name

    @abstractmethod
    def _load(self) -> T:
        """Load the resource.

        Subclasses implement this to define resource loading logic.

        Returns
        -------
        T
            The loaded resource.

        Raises
        ------
        Exception
            If loading fails for any reason.
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
        ResourceNotLoadedError
            If loading fails or previously failed.
        """
        if self._loaded and self._resource is not None:
            return self._resource

        if self._load_error is not None:
            raise ResourceNotLoadedError(self._name, str(self._load_error))

        try:
            self._resource = self._load()
        except Exception as e:
            self._load_error = e
            raise ResourceNotLoadedError(self._name, str(e)) from e
        else:
            self._loaded = True
            return self._resource

    def get_or_none(self) -> T | None:
        """Get the resource or None if unavailable.

        This method never raises exceptions. Use this when the resource
        is optional and you want to handle absence gracefully.

        Returns
        -------
        T | None
            The loaded resource, or None if loading fails.
        """
        try:
            return self.get()
        except ResourceNotLoadedError:
            return None

    def invalidate(self) -> None:
        """Invalidate the cached resource.

        Clears the cached resource, loaded flag, and any error state,
        forcing a fresh load on the next `get()` call.
        """
        self._resource = None
        self._loaded = False
        self._load_error = None

    def set_preloaded(self, resource: T) -> None:
        """Set a pre-loaded resource value.

        Use this to inject an already-loaded resource into the provider
        without triggering the ``_load()`` method. This is particularly
        useful for testing and dependency injection scenarios.

        Parameters
        ----------
        resource
            The pre-loaded resource value.
        """
        self._resource = resource
        self._loaded = True
        self._load_error = None


__all__ = [
    "LazyResource",
    "ResourceError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceProviderBase",
]
