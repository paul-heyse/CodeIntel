"""Resource container for dependency injection.

This module provides a graph-specific DI container that extends the unified
ResourceRegistry with factory support and provider-based registration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar, cast

from codeintel.core.resources import ResourceProvider
from codeintel.core.resources.registry import ResourceNotFoundError, ResourceRegistry

log = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceContainer(ResourceRegistry):
    """Graph-specific resource container extending unified registry.

    Extends ResourceRegistry with:
    - Provider-based registration using RESOURCE_NAME attribute
    - Factory support for lazy provider instantiation

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
        super().__init__()
        self._factories: dict[str, Callable[[], ResourceProvider[object]]] = {}

    def register(  # type: ignore[override]
        self,
        provider: ResourceProvider[object],
    ) -> None:
        """Register a resource provider using its RESOURCE_NAME.

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
        if self.has_by_name(name):
            log.warning("Overwriting resource provider: %s", name)
        self.register_by_name(name, provider)
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

    def _resolve_factory(self, name: str) -> bool:
        """Resolve a factory if the resource is not yet registered.

        Parameters
        ----------
        name
            Resource name to check.

        Returns
        -------
        bool
            True if a factory was resolved, False otherwise.
        """
        if not self.has_by_name(name) and name in self._factories:
            factory = self._factories[name]
            provider = factory()
            self.register_by_name(name, provider)
            return True
        return False

    def require(self, provider_type: type[ResourceProvider[T]]) -> T:  # type: ignore[override]
        """Get a resource by provider type.

        Delegates to :meth:`require_by_name` which raises ``ResourceNotFoundError``
        if no provider is registered for the type.

        Parameters
        ----------
        provider_type
            The provider class to look up.

        Returns
        -------
        T
            The resource value.
        """
        name = self._get_resource_name(provider_type)
        self._resolve_factory(name)
        return cast("T", self.require_by_name(name))

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
        self._resolve_factory(name)
        try:
            return super().require_by_name(name)
        except KeyError as exc:
            raise ResourceNotFoundError(name) from exc

    def get(self, name: str, default: object | None = None) -> object | None:  # type: ignore[override]
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

    def has(self, name: str) -> bool:  # type: ignore[override]
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
        return self.has_by_name(name) or name in self._factories

    def invalidate(self, name: str) -> None:  # type: ignore[override]
        """Invalidate a cached resource by name.

        Parameters
        ----------
        name
            Resource name to invalidate.
        """
        try:
            provider = self.get_by_name(name)
            if hasattr(provider, "invalidate"):
                provider.invalidate()  # type: ignore[union-attr]
        except KeyError:
            pass

    def invalidate_all(self) -> None:
        """Invalidate all cached resources."""
        for provider in self._by_name.values():
            if hasattr(provider, "invalidate"):
                provider.invalidate()  # type: ignore[union-attr]
        log.debug("Invalidated all resources")

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.invalidate_all()
        self.clear()
        self._factories.clear()

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
        for name in list(self._by_name.keys()):
            provider = self._by_name.get(name)
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
        return tuple(set(self._by_name.keys()) | set(self._factories.keys()))


__all__ = [
    "ResourceContainer",
    "ResourceNotFoundError",
]
