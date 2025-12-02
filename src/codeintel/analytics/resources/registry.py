"""Central registry for typed resource access.

This module provides `ResourceRegistry`, a container for managing
multiple resource providers with type-safe access.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from codeintel.analytics.resources.protocol import ResourceError, ResourceProvider

log = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")  # Key type (usually provider class)


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

    Providers can also be accessed by string name, enabling TYPE_CHECKING
    imports without runtime circular dependencies.

    Example
    -------
    >>> from codeintel.analytics.resources import ResourceRegistry, GraphProvider
    >>> registry = ResourceRegistry()
    >>> registry.register(GraphProvider, graph_provider)
    >>> provider = registry.get(GraphProvider)
    >>> call_graph = provider.call_graph
    >>> # String-based lookup (avoids import cycles):
    >>> same_provider = registry.get_by_name("GraphProvider")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: dict[type, ResourceProvider[Any]] = {}
        self._providers_by_name: dict[str, ResourceProvider[Any]] = {}

    def register(
        self,
        resource_type: type[K],
        provider: ResourceProvider[Any],
        *,
        name: str | None = None,
    ) -> None:
        """Register a resource provider.

        The resource_type is used as a lookup key and does not need to match
        the provider's generic type parameter. This allows registering by
        provider class (e.g., GraphProvider) while the provider returns
        a different type (e.g., GraphResources).

        The provider is also registered by string name (defaults to the
        class name) to support TYPE_CHECKING imports without runtime
        circular dependencies.

        Parameters
        ----------
        resource_type
            The type key for this provider (typically the provider class).
        provider
            The resource provider instance.
        name
            Optional explicit string name. Defaults to resource_type.__name__.

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

        # Also register by string name
        string_name = name if name is not None else resource_type.__name__
        self._providers_by_name[string_name] = provider

        log.debug("Registered resource provider: %s", resource_type.__name__)

    def register_or_replace(
        self,
        resource_type: type[K],
        provider: ResourceProvider[Any],
        *,
        name: str | None = None,
    ) -> ResourceProvider[Any] | None:
        """Register or replace a resource provider.

        Parameters
        ----------
        resource_type
            The type key for this provider (typically the provider class).
        provider
            The resource provider instance.
        name
            Optional explicit string name. Defaults to resource_type.__name__.

        Returns
        -------
        ResourceProvider[Any] | None
            The previous provider if one was replaced, None otherwise.
        """
        previous = self._providers.get(resource_type)
        self._providers[resource_type] = provider

        # Also register by string name
        string_name = name if name is not None else resource_type.__name__
        self._providers_by_name[string_name] = provider

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

    def get(self, resource_type: type[K]) -> ResourceProvider[Any]:
        """Get a resource provider by type key.

        Parameters
        ----------
        resource_type
            The type key for the provider (typically the provider class).

        Returns
        -------
        ResourceProvider[Any]
            The registered provider.

        Raises
        ------
        ResourceNotFoundError
            If no provider is registered for the type.
        """
        provider = self._providers.get(resource_type)
        if provider is None:
            raise ResourceNotFoundError(resource_type)
        return provider

    def get_or_none(self, resource_type: type[K]) -> ResourceProvider[Any] | None:
        """Get a resource provider or None if not registered.

        Parameters
        ----------
        resource_type
            The type key for the provider (typically the provider class).

        Returns
        -------
        ResourceProvider[Any] | None
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

    def get_by_name(self, name: str) -> ResourceProvider[Any]:
        """Get a resource provider by string name.

        Use this for TYPE_CHECKING imports to avoid circular dependencies.

        Parameters
        ----------
        name
            The string name of the provider (typically the class name).

        Returns
        -------
        ResourceProvider[Any]
            The registered provider.

        Raises
        ------
        KeyError
            If no provider is registered with that name.
        """
        provider = self._providers_by_name.get(name)
        if provider is None:
            message = f"Resource not found by name: {name}"
            raise KeyError(message)
        return provider

    def has_by_name(self, name: str) -> bool:
        """Check if a provider is registered by string name.

        Parameters
        ----------
        name
            The string name to check.

        Returns
        -------
        bool
            True if a provider is registered with that name.
        """
        return name in self._providers_by_name

    def require_by_name(self, name: str) -> object:
        """Get the resource value by name, loading if necessary.

        Convenience method that gets the provider by name and calls `.get()`.
        Use this for TYPE_CHECKING imports to avoid circular dependencies.

        Parameters
        ----------
        name
            The string name of the provider (typically the class name).

        Returns
        -------
        object
            The loaded resource. Caller should cast to the expected type.

        Notes
        -----
        Raises ``KeyError`` (via ``get_by_name``) if no provider is registered.
        """
        provider = self.get_by_name(name)
        return provider.get()

    def require(self, resource_type: type[T]) -> T:
        """Get the resource value, loading if necessary.

        Convenience method that gets the provider and calls `.get()`.

        Parameters
        ----------
        resource_type
            The type key for the provider (typically the provider class).

        Returns
        -------
        T
            The loaded resource. For type safety, cast the result to the
            expected resource type at the call site.

        Notes
        -----
        May raise `ResourceNotFoundError` if no provider is registered,
        or `ResourceNotLoadedError` if the resource cannot be loaded.
        """
        provider = self.get(resource_type)
        return cast("T", provider.get())

    def require_or_none(self, resource_type: type[T]) -> T | None:
        """Get the resource value or None if unavailable.

        Parameters
        ----------
        resource_type
            The type key for the provider (typically the provider class).

        Returns
        -------
        T | None
            The loaded resource, or None if unavailable.
        """
        provider = self.get_or_none(resource_type)
        if provider is None:
            return None
        result = provider.get_or_none()
        return cast("T | None", result)

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
        self._providers_by_name.clear()
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
