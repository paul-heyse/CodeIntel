"""Unified resource registry.

This module provides a registry for managing resource providers that
works with both graph and analytics plugins.

The registry supports both type-based and string-based lookup, enabling
TYPE_CHECKING imports without runtime circular dependencies. It also
supports factory registration for lazy provider instantiation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeGuard, TypeVar, cast, runtime_checkable

from codeintel.core.resources.protocol import ResourceError

log = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")


# =============================================================================
# Type Guards for Provider Protocol
# =============================================================================


@runtime_checkable
class _Gettable(Protocol):
    """Protocol for objects with a get() method."""

    def get(self) -> object:
        """Return the resource value."""
        ...


@runtime_checkable
class _Invalidatable(Protocol):
    """Protocol for objects with an invalidate() method."""

    def invalidate(self) -> None:
        """Invalidate the cached resource."""
        ...


def _is_gettable(obj: object) -> TypeGuard[_Gettable]:
    """Type guard to check if object has a callable get() method.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    TypeGuard[_Gettable]
        True if object implements get().
    """
    return isinstance(obj, _Gettable)


def _is_invalidatable(obj: object) -> TypeGuard[_Invalidatable]:
    """Type guard to check if object has a callable invalidate() method.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    TypeGuard[_Invalidatable]
        True if object implements invalidate().
    """
    return isinstance(obj, _Invalidatable)


class ResourceNotFoundError(ResourceError):
    """Raised when a required resource is not available."""

    def __init__(self, resource_type_or_name: type | str) -> None:
        """Initialize the error.

        Parameters
        ----------
        resource_type_or_name
            Either a type (class) or a string name of the missing resource.
        """
        if isinstance(resource_type_or_name, type):
            self.resource_type: type | None = resource_type_or_name
            self.resource_name = resource_type_or_name.__name__
        else:
            self.resource_type = None
            self.resource_name = resource_type_or_name
        super().__init__(f"Resource not found: {self.resource_name}")


class ResourceRegistry:
    """Registry for typed resource providers.

    This registry provides a clean interface for plugins to access
    resources without direct dependencies between plugins. It supports
    lookup by type or by string name.

    The registry distinguishes between:
    - `get()` / `get_or_none()`: Return the provider itself
    - `require()` / `require_or_none()`: Return the provider's value (calls `.get()`)

    Example
    -------
    >>> from codeintel.core.resources import ResourceRegistry
    >>> registry = ResourceRegistry()
    >>> registry.register(GraphProvider, graph_provider)
    >>> provider = registry.get(GraphProvider)  # returns provider
    >>> value = registry.require(GraphProvider)  # returns provider.get()
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._providers: dict[type[Any], object] = {}
        self._by_name: dict[str, object] = {}
        self._factories: dict[str, Callable[[], object]] = {}

    def register(
        self,
        resource_type: type[K],
        provider: object,
        *,
        name: str | None = None,
    ) -> None:
        """Register a resource provider.

        The resource_type is used as a lookup key and does not need to match
        the provider's generic type parameter. This allows registering by
        provider class while the provider returns a different type.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.
        name
            Optional explicit string name. Defaults to resource_type.__name__.

        Raises
        ------
        ValueError
            If a provider is already registered for this type.
        """
        if resource_type in self._providers:
            existing = self._providers[resource_type]
            existing_name = getattr(existing, "RESOURCE_NAME", "") or resource_type.__name__
            message = (
                f"Resource {resource_type.__name__} already registered "
                f"with provider {existing_name}"
            )
            raise ValueError(message)

        self._providers[resource_type] = provider

        # Also register by string name
        string_name = name if name is not None else resource_type.__name__
        self._by_name[string_name] = provider

        log.debug("Registered resource provider: %s", resource_type.__name__)

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
        log.debug("Registered resource provider by name: %s", name)

    def register_or_replace(
        self,
        resource_type: type[K],
        provider: object,
        *,
        name: str | None = None,
    ) -> object | None:
        """Register or replace a resource provider.

        Use this for testing scenarios where you need to override providers.

        Parameters
        ----------
        resource_type
            Type key for the provider.
        provider
            Resource provider instance.
        name
            Optional explicit string name. Defaults to resource_type.__name__.

        Returns
        -------
        object | None
            The previous provider if one was replaced, None otherwise.
        """
        previous = self._providers.get(resource_type)
        self._providers[resource_type] = provider

        # Also register by string name
        string_name = name if name is not None else resource_type.__name__
        self._by_name[string_name] = provider

        if previous:
            prev_name = getattr(previous, "RESOURCE_NAME", "") or resource_type.__name__
            new_name = getattr(provider, "RESOURCE_NAME", "") or resource_type.__name__
            log.debug(
                "Replaced resource provider %s: %s -> %s",
                resource_type.__name__,
                prev_name,
                new_name,
            )
        else:
            log.debug("Registered resource provider: %s", resource_type.__name__)
        return previous

    def register_provider(self, provider: object) -> None:
        """Register a resource provider using its RESOURCE_NAME attribute.

        Convenience method for providers that define a RESOURCE_NAME class attribute.
        The provider is registered by name only (not by type).

        Parameters
        ----------
        provider
            Resource provider instance with a RESOURCE_NAME attribute.

        Raises
        ------
        ValueError
            If the provider does not have a RESOURCE_NAME attribute.
        """
        name = getattr(type(provider), "RESOURCE_NAME", None)
        if name is None:
            message = f"Provider {type(provider).__name__} has no RESOURCE_NAME attribute"
            raise ValueError(message)
        if self.has_by_name(name):
            log.warning("Overwriting resource provider: %s", name)
        self.register_by_name(name, provider)
        log.debug("Registered resource provider: %s", name)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> None:
        """Register a factory for lazy provider creation.

        Factories are called lazily when the resource is first requested.
        Once instantiated, the provider is cached and the factory is not
        called again.

        Parameters
        ----------
        name
            Resource name for lookup.
        factory
            Factory function that creates the provider.
        """
        self._factories[name] = factory
        log.debug("Registered resource factory: %s", name)

    def _resolve_factory(self, name: str) -> bool:
        """Resolve a factory if the resource is not yet instantiated.

        Parameters
        ----------
        name
            Resource name to check.

        Returns
        -------
        bool
            True if a factory was resolved, False otherwise.
        """
        # Check if provider is already instantiated (in _by_name)
        # If not, but a factory exists, instantiate it
        if name not in self._by_name and name in self._factories:
            factory = self._factories[name]
            provider = factory()
            self.register_by_name(name, provider)
            return True
        return False

    def get(self, resource_type: type[K]) -> object:
        """Get a resource provider by type key.

        Returns the provider itself, not its value. Use `require()` to get
        the provider's value.

        Parameters
        ----------
        resource_type
            Type key for the provider.

        Returns
        -------
        object
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

    def get_or_none(self, resource_type: type[K]) -> object | None:
        """Get a resource provider or None if not registered.

        Returns the provider itself, not its value.

        Parameters
        ----------
        resource_type
            Type key for the provider.

        Returns
        -------
        object | None
            The registered provider, or None if not registered.
        """
        return self._providers.get(resource_type)

    def get_by_name(self, name: str) -> object:
        """Get a resource provider by string name.

        Use this for TYPE_CHECKING imports to avoid circular dependencies.
        If the resource is not yet instantiated but a factory exists,
        the factory is called first.

        Parameters
        ----------
        name
            String name of the provider (typically the class name).

        Returns
        -------
        object
            The registered provider.

        Raises
        ------
        KeyError
            If no provider is registered with that name.
        """
        self._resolve_factory(name)
        provider = self._by_name.get(name)
        if provider is None:
            message = f"Resource not found by name: {name}"
            raise KeyError(message)
        return provider

    def require(self, resource_type: type[T]) -> T:
        """Get the resource value, loading if necessary.

        Calls `.get()` on the provider to retrieve the actual resource value.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T
            The resource value (result of provider.get()).

        Raises
        ------
        ResourceNotFoundError
            If the resource is not registered.
        """
        if resource_type not in self._providers:
            raise ResourceNotFoundError(resource_type)
        provider = self._providers[resource_type]
        # Call .get() if the provider has it, otherwise return the provider itself
        if _is_gettable(provider):
            return cast("T", provider.get())
        return cast("T", provider)

    def require_or_none(self, resource_type: type[T]) -> T | None:
        """Get the resource value or None if unavailable.

        This method never raises for resource-related errors. If the resource
        is not registered or cannot be loaded, None is returned.

        Parameters
        ----------
        resource_type
            Type of resource to retrieve.

        Returns
        -------
        T | None
            The resource value, or None if unavailable.
        """
        provider = self._providers.get(resource_type)
        if provider is None:
            return None
        # Call .get() if available, with error handling
        if _is_gettable(provider):
            try:
                return cast("T", provider.get())
            except ResourceError:
                return None
        return cast("T | None", provider)

    def require_by_name(self, name: str) -> object:
        """Get the resource value by name, loading if necessary.

        Convenience method that gets the provider by name and calls `.get()`.

        Parameters
        ----------
        name
            String name of the provider (typically the class name).

        Returns
        -------
        object
            The loaded resource value.

        """
        provider = self.get_by_name(name)
        if _is_gettable(provider):
            return provider.get()
        return provider

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
        """Check if a resource is registered or has a factory by string name.

        Parameters
        ----------
        name
            String name to check.

        Returns
        -------
        bool
            True if a resource with that name is available or can be created.
        """
        return name in self._by_name or name in self._factories

    def invalidate(self, resource_type: type | None = None) -> None:
        """Invalidate cached resources.

        Calls `.invalidate()` on providers to force a reload on next access.

        Parameters
        ----------
        resource_type
            If provided, invalidate only this resource type.
            If None, invalidate all resources (both type-keyed and name-keyed).
        """
        if resource_type is not None:
            provider = self._providers.get(resource_type)
            if provider is not None and _is_invalidatable(provider):
                provider.invalidate()
                log.debug("Invalidated resource: %s", resource_type.__name__)
        else:
            # Collect all unique providers from both dicts (using id() for uniqueness)
            seen_ids: set[int] = set()
            for provider in list(self._providers.values()) + list(self._by_name.values()):
                provider_id = id(provider)
                if provider_id not in seen_ids:
                    seen_ids.add(provider_id)
                    if _is_invalidatable(provider):
                        provider.invalidate()
            log.debug("Invalidated all resources")

    def clear(self) -> None:
        """Clear all registered providers and factories."""
        self._providers.clear()
        self._by_name.clear()
        self._factories.clear()
        log.debug("Cleared resource registry")

    def cleanup(self) -> None:
        """Invalidate all resources and clear the registry.

        Combines `invalidate()` and `clear()` for complete cleanup.
        Useful for test teardown or resource lifecycle management.
        """
        self.invalidate()
        self.clear()
        log.debug("Cleaned up resource registry")

    @property
    def registered_names(self) -> tuple[str, ...]:
        """Return all registered resource names including pending factories.

        Returns
        -------
        tuple[str, ...]
            All registered names and factory names.
        """
        return tuple(sorted(set(self._by_name.keys()) | set(self._factories.keys())))

    @property
    def registered_types(self) -> frozenset[type]:
        """Return the set of registered resource types.

        Returns
        -------
        frozenset[type]
            All registered type keys.
        """
        return frozenset(self._providers.keys())

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
