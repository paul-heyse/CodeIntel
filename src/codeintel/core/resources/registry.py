"""Unified resource registry.

This module provides a registry for managing resource providers that
works with both graph and analytics plugins.

The registry supports both type-based and string-based lookup, enabling
TYPE_CHECKING imports without runtime circular dependencies. It also
supports factory registration for lazy provider instantiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

from codeintel.core.resources.protocol import ResourceError

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.core.resources.protocol import LazyResource
log = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
R = TypeVar("R")


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


@dataclass
class ResourceEntry:
    """Typed resource entry with optional lazy factory."""

    name: str
    provider: object | None
    resource_type: type[Any] | None = None
    factory: Callable[[], object] | None = None

    @property
    def is_factory(self) -> bool:
        """Return True when this entry uses a lazy factory."""
        return self.factory is not None and self.provider is None


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
    >>> from codeintel.core.catalog import CatalogService
    >>> registry = ResourceRegistry()
    >>> registry.register(CatalogService, CatalogService(catalog))
    >>> provider = registry.get(CatalogService)
    >>> value = registry.require(CatalogService)
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._entries_by_type: dict[type[Any], ResourceEntry] = {}
        self._entries_by_name: dict[str, ResourceEntry] = {}

    def register_singleton(
        self,
        resource_type: type[K],
        provider: object,
        *,
        name: str | None = None,
    ) -> ResourceEntry:
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

        Returns
        -------
        ResourceEntry
            Registry entry created for the provider.
        """
        string_name = name if name is not None else resource_type.__name__

        if resource_type in self._entries_by_type:
            existing = self._entries_by_type[resource_type].provider
            existing_name = getattr(existing, "RESOURCE_NAME", "") or resource_type.__name__
            message = (
                f"Resource {resource_type.__name__} already registered "
                f"with provider {existing_name}"
            )
            raise ValueError(message)

        if string_name in self._entries_by_name:
            message = f"Resource name already registered: {string_name}"
            raise ValueError(message)

        entry = ResourceEntry(
            name=string_name,
            provider=provider,
            resource_type=resource_type,
        )
        self._entries_by_type[resource_type] = entry
        self._entries_by_name[string_name] = entry

        log.debug("Registered resource provider: %s", string_name)
        return entry

    register = register_singleton

    def register_by_name(
        self, name: str, provider: object, *, allow_overwrite: bool = False
    ) -> None:
        """Register a resource provider by string name only.

        Parameters
        ----------
        name
            String name for the provider.
        provider
            Resource provider instance.
        allow_overwrite
            When True, replace any existing provider registered under the
            same name (useful for tests or dynamic reloads). When False,
            raises if the name is already registered.

        Raises
        ------
        ValueError
            If the name is already registered.
        """
        existing_entry = self._entries_by_name.get(name)
        if existing_entry is not None and not allow_overwrite:
            message = f"Resource name already registered: {name}"
            raise ValueError(message)
        entry = ResourceEntry(
            name=name,
            provider=provider,
            resource_type=existing_entry.resource_type if existing_entry else None,
        )
        if entry.resource_type is not None:
            self._entries_by_type[entry.resource_type] = entry
        self._entries_by_name[name] = entry
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
        string_name = name if name is not None else resource_type.__name__
        previous_entry = self._entries_by_type.get(resource_type)
        entry = ResourceEntry(
            name=string_name,
            provider=provider,
            resource_type=resource_type,
        )
        self._entries_by_type[resource_type] = entry
        self._entries_by_name[string_name] = entry

        if previous_entry is not None:
            prev_name = getattr(previous_entry.provider, "RESOURCE_NAME", "") or previous_entry.name
            new_name = getattr(provider, "RESOURCE_NAME", "") or resource_type.__name__
            log.debug(
                "Replaced resource provider %s: %s -> %s",
                resource_type.__name__,
                prev_name,
                new_name,
            )
            return previous_entry.provider

        log.debug("Registered resource provider: %s", resource_type.__name__)
        return None

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
        self.register_by_name(name, provider, allow_overwrite=True)
        log.debug("Registered resource provider: %s", name)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
        *,
        resource_type: type[Any] | None = None,
    ) -> ResourceEntry:
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
        resource_type
            Optional type key to index the factory under.

        Returns
        -------
        ResourceEntry
            Entry created for the factory registration.
        """
        entry = ResourceEntry(
            name=name,
            provider=None,
            resource_type=resource_type,
            factory=factory,
        )
        self._entries_by_name[name] = entry
        if resource_type is not None:
            self._entries_by_type[resource_type] = entry
        log.debug("Registered resource factory: %s", name)
        return entry

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
        entry = self._entries_by_name.get(name)
        if entry is None or entry.provider is not None or entry.factory is None:
            return False
        provider = entry.factory()
        entry.provider = provider
        if entry.resource_type is not None:
            self._entries_by_type[entry.resource_type] = entry
        log.debug("Materialized resource provider from factory: %s", name)
        return True

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
        entry = self._entries_by_type.get(resource_type)
        if entry is None:
            raise ResourceNotFoundError(resource_type)
        self._resolve_factory(entry.name)
        entry = self._entries_by_type.get(resource_type)
        if entry is None or entry.provider is None:
            raise ResourceNotFoundError(resource_type)
        return entry.provider

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
        entry = self._entries_by_type.get(resource_type)
        if entry is None:
            return None
        self._resolve_factory(entry.name)
        refreshed = self._entries_by_type.get(resource_type)
        return refreshed.provider if refreshed is not None else None

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
        entry = self._entries_by_name.get(name)
        provider = entry.provider if entry is not None else None
        if provider is None:
            message = f"Resource not found by name: {name}"
            raise KeyError(message)
        return provider

    @overload
    def require(self, resource_type: type[LazyResource[R]]) -> R: ...

    @overload
    def require(self, resource_type: type[T]) -> T: ...

    def require(self, resource_type: type[object]) -> object:
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
        entry = self._entries_by_type.get(resource_type)
        if entry is None:
            raise ResourceNotFoundError(resource_type)
        self._resolve_factory(entry.name)
        entry = self._entries_by_type.get(resource_type)
        if entry is None or entry.provider is None:
            raise ResourceNotFoundError(resource_type)
        provider = entry.provider

        if _is_gettable(provider):
            return provider.get()
        return provider

    @overload
    def require_or_none(self, resource_type: type[LazyResource[R]]) -> R | None: ...

    @overload
    def require_or_none(self, resource_type: type[T]) -> T | None: ...

    def require_or_none(self, resource_type: type[object]) -> object | None:
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
        entry = self._entries_by_type.get(resource_type)
        if entry is None:
            return None
        self._resolve_factory(entry.name)
        entry = self._entries_by_type.get(resource_type)
        if entry is None or entry.provider is None:
            return None
        provider = entry.provider

        if _is_gettable(provider):
            try:
                return provider.get()
            except ResourceError:
                return None
        return provider

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
        return resource_type in self._entries_by_type

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
        entry = self._entries_by_name.get(name)
        return entry is not None and (entry.provider is not None or entry.factory is not None)

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
            entry = self._entries_by_type.get(resource_type)
            if (
                entry is not None
                and entry.provider is not None
                and _is_invalidatable(entry.provider)
            ):
                entry.provider.invalidate()
                log.debug("Invalidated resource: %s", resource_type.__name__)
            return

        seen_ids: set[int] = set()
        for entry in self._entries_by_name.values():
            provider = entry.provider
            if provider is None:
                continue
            provider_id = id(provider)
            if provider_id not in seen_ids:
                seen_ids.add(provider_id)
                if _is_invalidatable(provider):
                    provider.invalidate()
        log.debug("Invalidated all resources")

    def clear(self) -> None:
        """Clear all registered providers and factories."""
        self._entries_by_type.clear()
        self._entries_by_name.clear()
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
        return tuple(sorted(self._entries_by_name.keys()))

    @property
    def registered_types(self) -> frozenset[type]:
        """Return the set of registered resource types.

        Returns
        -------
        frozenset[type]
            All registered type keys.
        """
        return frozenset(self._entries_by_type.keys())

    def __len__(self) -> int:
        """Return the number of registered providers.

        Returns
        -------
        int
            Count of registered providers.
        """
        return len(self._entries_by_type)

    __contains__ = has


__all__ = [
    "ResourceEntry",
    "ResourceNotFoundError",
    "ResourceRegistry",
]
