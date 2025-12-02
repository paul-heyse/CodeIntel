"""Central registry for typed resource access.

This module provides `ResourceRegistry`, a container for managing
multiple resource providers with type-safe access.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from codeintel.analytics.resources.protocol import ResourceError, ResourceProvider

log = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceNotFoundError(ResourceError):
    """Requested resource type is not registered."""

    def __init__(self, resource_type: type) -> None:
        """Initialize the error.

        Parameters
        ----------
        resource_type
            The resource type that was not found.
        """
        super().__init__(f"Resource not found: {resource_type.__name__}")
        self.resource_type = resource_type


class ResourceRegistry:
    """Central registry for lazy resource access.

    The registry provides typed access to resource providers. Resources
    are registered by type and retrieved using the same type key.

    Example
    -------
    >>> from codeintel.analytics.resources import ResourceRegistry, GraphProvider
    >>> registry = ResourceRegistry()
    >>> registry.register(GraphProvider, graph_provider)
    >>> provider = registry.get(GraphProvider)
    >>> call_graph = provider.call_graph
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: dict[type, ResourceProvider[Any]] = {}

    def register(self, resource_type: type[T], provider: ResourceProvider[T]) -> None:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            The type key for this provider.
        provider
            The resource provider instance.

        Raises
        ------
        ValueError
            If a provider is already registered for this type.
        """
        if resource_type in self._providers:
            existing = self._providers[resource_type]
            message = (
                f"Resource {resource_type.__name__} already registered "
                f"with provider {existing.resource_name}"
            )
            raise ValueError(message)

        self._providers[resource_type] = provider
        log.debug("Registered resource provider: %s", resource_type.__name__)

    def register_or_replace(
        self, resource_type: type[T], provider: ResourceProvider[T]
    ) -> ResourceProvider[T] | None:
        """Register or replace a resource provider.

        Parameters
        ----------
        resource_type
            The type key for this provider.
        provider
            The resource provider instance.

        Returns
        -------
        ResourceProvider[T] | None
            The previous provider if one was replaced, None otherwise.
        """
        previous = self._providers.get(resource_type)
        self._providers[resource_type] = provider
        if previous:
            log.debug(
                "Replaced resource provider %s: %s -> %s",
                resource_type.__name__,
                previous.resource_name,
                provider.resource_name,
            )
        else:
            log.debug("Registered resource provider: %s", resource_type.__name__)
        return previous

    def get(self, resource_type: type[T]) -> ResourceProvider[T]:
        """Get a resource provider by type.

        Parameters
        ----------
        resource_type
            The type key for the provider.

        Returns
        -------
        ResourceProvider[T]
            The registered provider.

        Raises
        ------
        ResourceNotFoundError
            If no provider is registered for the type.
        """
        provider = self._providers.get(resource_type)
        if provider is None:
            raise ResourceNotFoundError(resource_type)
        # Type is guaranteed by registration
        return provider  # type: ignore[return-value]

    def get_or_none(self, resource_type: type[T]) -> ResourceProvider[T] | None:
        """Get a resource provider or None if not registered.

        Parameters
        ----------
        resource_type
            The type key for the provider.

        Returns
        -------
        ResourceProvider[T] | None
            The registered provider, or None if not registered.
        """
        try:
            return self.get(resource_type)
        except ResourceNotFoundError:
            return None

    def has(self, resource_type: type) -> bool:
        """Check if a resource type is registered.

        Parameters
        ----------
        resource_type
            The type to check.

        Returns
        -------
        bool
            True if a provider is registered for the type.
        """
        return resource_type in self._providers

    def require(self, resource_type: type[T]) -> T:
        """Get the resource value, loading if necessary.

        Convenience method that gets the provider and calls `.get()`.

        Parameters
        ----------
        resource_type
            The type key for the provider.

        Returns
        -------
        T
            The loaded resource.

        Raises
        ------
        ResourceNotFoundError
            If no provider is registered.
        ResourceNotLoadedError
            If the resource cannot be loaded.
        """
        provider = self.get(resource_type)
        return provider.get()

    def require_or_none(self, resource_type: type[T]) -> T | None:
        """Get the resource value or None if unavailable.

        Parameters
        ----------
        resource_type
            The type key for the provider.

        Returns
        -------
        T | None
            The loaded resource, or None if unavailable.
        """
        provider = self.get_or_none(resource_type)
        if provider is None:
            return None
        return provider.get_or_none()

    def invalidate(self, resource_type: type | None = None) -> None:
        """Invalidate cached resources.

        Parameters
        ----------
        resource_type
            If provided, invalidate only this resource type.
            If None, invalidate all resources.
        """
        if resource_type is not None:
            provider = self._providers.get(resource_type)
            if provider is not None:
                provider.invalidate()
                log.debug("Invalidated resource: %s", resource_type.__name__)
        else:
            for provider in self._providers.values():
                provider.invalidate()
            log.debug("Invalidated all resources")

    def clear(self) -> None:
        """Remove all registered providers."""
        self._providers.clear()
        log.debug("Cleared resource registry")

    @property
    def registered_types(self) -> frozenset[type]:
        """Return the set of registered resource types.

        Returns
        -------
        frozenset[type]
            All registered type keys.
        """
        return frozenset(self._providers.keys())


__all__ = [
    "ResourceNotFoundError",
    "ResourceRegistry",
]
