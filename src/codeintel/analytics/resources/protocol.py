"""Protocol and base classes for resource providers.

This module re-exports unified resource provider types from codeintel.core.resources,
while maintaining backward compatibility with analytics-specific resource patterns.

The canonical protocol definition lives in codeintel.core.resources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from codeintel.core.resources import (
    ResourceNotFoundError,
    ResourceProvider,
    ResourceProviderBase,
    ResourceRegistry,
)

T = TypeVar("T")


class ResourceError(Exception):
    """Base exception for resource-related errors."""


class ResourceNotLoadedError(ResourceError):
    """Resource has not been loaded yet."""

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


class LazyResource[T](ABC):
    """Abstract base class for lazy resource providers.

    Provides a standard implementation of lazy loading with caching and
    error tracking. Subclasses implement `_load()` to define how the
    resource is loaded.

    This class provides additional features beyond ResourceProviderBase:
    - is_loaded property to check load status
    - get_or_none() for optional access
    - set_preloaded() for dependency injection
    - Error tracking for repeated failures

    Type Parameters
    ---------------
    T
        The type of resource this provider manages.
    """

    def __init__(self, name: str) -> None:
        """Initialize the lazy resource.

        Parameters
        ----------
        name
            Human-readable name for the resource.
        """
        self._name = name
        self._resource: T | None = None
        self._loaded = False
        self._load_error: Exception | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if the resource has been loaded."""
        return self._loaded

    @property
    def resource_name(self) -> str:
        """Return the resource name."""
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
        ResourceNotLoadedError
            If loading fails.
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
        """Invalidate the cached resource."""
        self._resource = None
        self._loaded = False
        self._load_error = None

    def set_preloaded(self, resource: T) -> None:
        """Set a pre-loaded resource value.

        Use this to inject an already-loaded resource into the provider
        without triggering the `_load()` method.

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
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
]
