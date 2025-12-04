"""Unified resource registry.

This module provides a registry for managing resource providers that
works with both graph and analytics plugins.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

T = TypeVar("T")


class ResourceNotFoundError(Exception):
    """Raised when a required resource is not available."""

    def __init__(self, resource_name: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        resource_name
            Name of the missing resource.
        """
        super().__init__(f"Resource not found: {resource_name}")
        self.resource_name = resource_name


class ResourceRegistry:
    """Registry for typed resource providers.

    This registry provides a clean interface for plugins to access
    resources without direct dependencies between plugins. It supports
    lookup by type or by string name.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: dict[type[Any], object] = {}
        self._by_name: dict[str, object] = {}

    def register(self, resource_type: type[T], provider: object) -> None:
        """Register a resource provider.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.
        """
        self._providers[resource_type] = provider
        self._by_name[resource_type.__name__] = provider

    def register_by_name(self, name: str, provider: object) -> None:
        """Register a resource provider by string name only.

        Parameters
        ----------
        name
            String name for the provider.
        provider
            Resource provider instance.
        """
        self._by_name[name] = provider

    def require(self, resource_type: type[T]) -> T:
        """Get a resource from the registry.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T
            The resource provider.

        Raises
        ------
        ResourceNotFoundError
            If the resource is not registered.
        """
        if resource_type not in self._providers:
            raise ResourceNotFoundError(resource_type.__name__)
        return cast("T", self._providers[resource_type])

    def require_or_none(self, resource_type: type[T]) -> T | None:
        """Get a resource or None if unavailable.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T | None
            The resource, or None if unavailable.
        """
        return cast("T | None", self._providers.get(resource_type))

    def require_by_name(self, name: str) -> object:
        """Get a resource by string name.

        Parameters
        ----------
        name
            String name of the provider (typically the class name).

        Returns
        -------
        object
            The resource provider.

        Raises
        ------
        ResourceNotFoundError
            If the resource is not registered.
        """
        if name not in self._by_name:
            raise ResourceNotFoundError(name)
        return self._by_name[name]

    def has(self, resource_type: type) -> bool:
        """Check if a resource type is registered.

        Parameters
        ----------
        resource_type
            Type to check.

        Returns
        -------
        bool
            True if the resource is available.
        """
        return resource_type in self._providers

    def has_by_name(self, name: str) -> bool:
        """Check if a resource is registered by string name.

        Parameters
        ----------
        name
            String name to check.

        Returns
        -------
        bool
            True if a resource with that name is available.
        """
        return name in self._by_name

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._by_name.clear()

    def __len__(self) -> int:
        """Return the number of registered providers.

        Returns
        -------
        int
            Count of registered providers.
        """
        return len(self._providers)

    def __contains__(self, resource_type: type) -> bool:
        """Check if a resource type is registered.

        Parameters
        ----------
        resource_type
            Type to check.

        Returns
        -------
        bool
            True if the resource is available.
        """
        return self.has(resource_type)


__all__ = [
    "ResourceNotFoundError",
    "ResourceRegistry",
]
