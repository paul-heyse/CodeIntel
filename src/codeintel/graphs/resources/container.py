"""Resource container for dependency injection.

This module provides a type-safe DI container for managing resource
providers in graph plugin execution.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar, cast

from codeintel.graphs.resources.protocol import ResourceProvider

log = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceNotFoundError(KeyError):
    """Raised when a required resource is not registered."""

    def __init__(self, resource_name: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        resource_name
            Name of the missing resource.
        """
        super().__init__(f"Resource not found: {resource_name}")
        self.resource_name = resource_name


class ResourceContainer:
    """Container for managing resource providers.

    The container stores resource providers and provides type-safe
    access via the require() method.

    Example
    -------
    ```python
    container = ResourceContainer()
    container.register(CatalogResource(catalog))
    container.register(StorageResource(gateway, repo_root))

    # Later, in a plugin:
    catalog = container.require(CatalogResource)
    storage = container.require(StorageResource)
    ```
    """

    def __init__(self) -> None:
        """Initialize an empty container."""
        self._providers: dict[str, ResourceProvider[object]] = {}
        self._factories: dict[str, Callable[[], ResourceProvider[object]]] = {}

    def register(self, provider: ResourceProvider[object]) -> None:
        """Register a resource provider.

        Parameters
        ----------
        provider
            Resource provider to register.

        Notes
        -----
        If a provider with the same name already exists, it will be overwritten
        and a warning will be logged.
        """
        name = type(provider).RESOURCE_NAME
        if name in self._providers:
            log.warning("Overwriting resource provider: %s", name)
        self._providers[name] = provider
        log.debug("Registered resource provider: %s", name)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], ResourceProvider[object]],
    ) -> None:
        """Register a factory for lazy provider creation.

        Parameters
        ----------
        name
            Resource name.
        factory
            Factory function that creates the provider.
        """
        self._factories[name] = factory
        log.debug("Registered resource factory: %s", name)

    def require(self, provider_type: type[ResourceProvider[T]]) -> T:
        """Get a resource by provider type.

        Parameters
        ----------
        provider_type
            The provider class to look up.

        Returns
        -------
        T
            The resource value.

        Raises
        ------
        ResourceNotFoundError
            If no provider is registered for the type.
        """
        # Get the resource name from the provider type
        name = self._get_resource_name(provider_type)

        # Check if we need to create from factory
        if name not in self._providers and name in self._factories:
            factory = self._factories[name]
            self._providers[name] = factory()

        provider = self._providers.get(name)
        if provider is None:
            raise ResourceNotFoundError(name)

        # Get the resource value - the caller is responsible for type safety
        # based on the provider_type they pass in
        return cast("T", provider.get())

    def require_by_name(self, name: str) -> object:
        """Get a resource by name.

        Parameters
        ----------
        name
            Resource name to look up.

        Returns
        -------
        object
            The resource value.

        Raises
        ------
        ResourceNotFoundError
            If no provider is registered for the name.
        """
        # Check if we need to create from factory
        if name not in self._providers and name in self._factories:
            factory = self._factories[name]
            self._providers[name] = factory()

        provider = self._providers.get(name)
        if provider is None:
            raise ResourceNotFoundError(name)
        return provider.get()

    def get(self, name: str, default: object | None = None) -> object | None:
        """Return a resource if registered, otherwise a default.

        Parameters
        ----------
        name
            Resource name to look up.
        default
            Value to return when the resource is not registered.

        Returns
        -------
        object | None
            The resource value or the provided default when missing.
        """
        try:
            return self.require_by_name(name)
        except ResourceNotFoundError:
            return default

    def has(self, name: str) -> bool:
        """Check if a resource is registered.

        Parameters
        ----------
        name
            Resource name to check.

        Returns
        -------
        bool
            True if the resource is registered.
        """
        return name in self._providers or name in self._factories

    def invalidate(self, name: str) -> None:
        """Invalidate a cached resource.

        Parameters
        ----------
        name
            Resource name to invalidate.
        """
        provider = self._providers.get(name)
        if provider is not None:
            provider.invalidate()

    def invalidate_all(self) -> None:
        """Invalidate all cached resources."""
        for provider in self._providers.values():
            provider.invalidate()

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.invalidate_all()
        self._providers.clear()

    def _get_resource_name(self, provider_type: type[ResourceProvider[T]]) -> str:
        """Extract resource name from a provider type.

        Parameters
        ----------
        provider_type
            Provider class.

        Returns
        -------
        str
            Resource name.
        """
        # Check if the class has a class-level resource_name
        resource_name = getattr(provider_type, "RESOURCE_NAME", None)
        if resource_name is not None:
            return str(resource_name)

        # Check registered providers for matching type
        for name, provider in self._providers.items():
            if isinstance(provider, provider_type):
                return name

        # Fall back to class name
        return provider_type.__name__

    @property
    def registered_names(self) -> tuple[str, ...]:
        """Get all registered resource names.

        Returns
        -------
        tuple[str, ...]
            Registered resource names.
        """
        return tuple(set(self._providers.keys()) | set(self._factories.keys()))


__all__ = [
    "ResourceContainer",
    "ResourceNotFoundError",
]
