"""Service registry for dependency injection.

This module provides a registry for managing service instances,
supporting both singleton and factory-based service creation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from codeintel.core.services.base import ServiceError
from codeintel.core.services.protocol import ServiceProtocol

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceNotFoundError(ServiceError):
    """Raised when a requested service is not registered."""

    def __init__(self, service_type: type | str) -> None:
        """Initialize the error.

        Parameters
        ----------
        service_type
            The service type or name that was not found.
        """
        name = service_type.__name__ if isinstance(service_type, type) else service_type
        super().__init__("registry", f"Service not found: {name}")
        self.service_type = service_type


class ServiceAlreadyRegisteredError(ServiceError):
    """Raised when attempting to register a service that already exists."""

    def __init__(self, service_type: type | str) -> None:
        """Initialize the error.

        Parameters
        ----------
        service_type
            The service type or name that is already registered.
        """
        name = service_type.__name__ if isinstance(service_type, type) else service_type
        super().__init__("registry", f"Service already registered: {name}")
        self.service_type = service_type


@dataclass
class ServiceEntry:
    """Entry in the service registry.

    Attributes
    ----------
    service_type
        The type key for the service.
    instance
        Singleton instance, if created.
    factory
        Factory function to create instances.
    singleton
        Whether to use singleton pattern.
    """

    service_type: type[Any]
    instance: ServiceProtocol | None = None
    factory: Callable[[], ServiceProtocol] | None = None
    singleton: bool = True


@dataclass
class ServiceRegistry:
    """Registry for managing service instances.

    Supports singleton and factory-based service creation with
    type-safe access patterns.

    Examples
    --------
    >>> registry = ServiceRegistry()
    >>> registry.register_singleton(DatabaseService, db_instance)
    >>> db = registry.get(DatabaseService)

    >>> registry.register_factory(CacheService, lambda: CacheService())
    >>> cache = registry.get(CacheService)
    """

    _entries_by_type: dict[type[Any], ServiceEntry] = field(default_factory=dict)
    _entries_by_name: dict[str, ServiceEntry] = field(default_factory=dict)

    def register_singleton(
        self,
        service_type: type[T],
        instance: T,
        *,
        name: str | None = None,
        allow_replace: bool = False,
    ) -> None:
        """Register a singleton service instance.

        Parameters
        ----------
        service_type
            Type key for the service.
        instance
            Service instance.
        name
            Optional string name for the service.
        allow_replace
            If True, replace existing registration.

        Raises
        ------
        ServiceAlreadyRegisteredError
            If service is already registered and allow_replace is False.
        """
        if not allow_replace and service_type in self._entries_by_type:
            raise ServiceAlreadyRegisteredError(service_type)

        entry = ServiceEntry(
            service_type=service_type,
            instance=cast("ServiceProtocol", instance),
            singleton=True,
        )

        self._entries_by_type[service_type] = entry

        resolved_name = name or service_type.__name__
        self._entries_by_name[resolved_name] = entry

        log.debug("Registered singleton service: %s", resolved_name)

    def register_factory(
        self,
        service_type: type[T],
        factory: Callable[[], T],
        *,
        name: str | None = None,
        singleton: bool = True,
        allow_replace: bool = False,
    ) -> None:
        """Register a factory for creating service instances.

        Parameters
        ----------
        service_type
            Type key for the service.
        factory
            Factory function that creates service instances.
        name
            Optional string name for the service.
        singleton
            If True, cache the first instance. If False, create new each time.
        allow_replace
            If True, replace existing registration.

        Raises
        ------
        ServiceAlreadyRegisteredError
            If service is already registered and allow_replace is False.
        """
        if not allow_replace and service_type in self._entries_by_type:
            raise ServiceAlreadyRegisteredError(service_type)

        entry = ServiceEntry(
            service_type=service_type,
            factory=cast("Callable[[], ServiceProtocol]", factory),
            singleton=singleton,
        )

        self._entries_by_type[service_type] = entry

        resolved_name = name or service_type.__name__
        self._entries_by_name[resolved_name] = entry

        log.debug(
            "Registered %s factory: %s",
            "singleton" if singleton else "transient",
            resolved_name,
        )

    def get(self, service_type: type[T]) -> T:
        """Get a service by type.

        Parameters
        ----------
        service_type
            Type of service to retrieve.

        Returns
        -------
        T
            Service instance.

        Raises
        ------
        ServiceNotFoundError
            If service is not registered.
        """
        entry = self._entries_by_type.get(service_type)
        if entry is None:
            raise ServiceNotFoundError(service_type)

        return cast("T", self._resolve_entry(entry))

    def get_or_none(self, service_type: type[T]) -> T | None:
        """Get a service by type, or None if not registered.

        Parameters
        ----------
        service_type
            Type of service to retrieve.

        Returns
        -------
        T | None
            Service instance or None.
        """
        entry = self._entries_by_type.get(service_type)
        if entry is None:
            return None

        return cast("T | None", self._resolve_entry(entry))

    def get_by_name(self, name: str) -> ServiceProtocol:
        """Get a service by string name.

        Parameters
        ----------
        name
            Name of service to retrieve.

        Returns
        -------
        ServiceProtocol
            Service instance.

        Raises
        ------
        ServiceNotFoundError
            If service is not registered.
        """
        entry = self._entries_by_name.get(name)
        if entry is None:
            raise ServiceNotFoundError(name)

        return self._resolve_entry(entry)

    @staticmethod
    def _resolve_entry(entry: ServiceEntry) -> ServiceProtocol:
        """Resolve a service entry to an instance.

        Parameters
        ----------
        entry
            Service registry entry.

        Returns
        -------
        ServiceProtocol
            Service instance.

        Raises
        ------
        ServiceNotFoundError
            If entry has no instance or factory.
        """
        if entry.instance is not None:
            return entry.instance

        if entry.factory is None:
            raise ServiceNotFoundError(entry.service_type)

        instance = entry.factory()

        if entry.singleton:
            entry.instance = instance

        return instance

    def has(self, service_type: type) -> bool:
        """Check if a service type is registered.

        Parameters
        ----------
        service_type
            Type to check.

        Returns
        -------
        bool
            True if registered.
        """
        return service_type in self._entries_by_type

    def has_by_name(self, name: str) -> bool:
        """Check if a service name is registered.

        Parameters
        ----------
        name
            Name to check.

        Returns
        -------
        bool
            True if registered.
        """
        return name in self._entries_by_name

    def unregister(self, service_type: type) -> bool:
        """Remove a service registration.

        Parameters
        ----------
        service_type
            Type of service to unregister.

        Returns
        -------
        bool
            True if service was found and removed.
        """
        entry = self._entries_by_type.pop(service_type, None)
        if entry is None:
            return False

        name = service_type.__name__
        self._entries_by_name.pop(name, None)

        log.debug("Unregistered service: %s", name)
        return True

    def clear(self) -> None:
        """Remove all service registrations."""
        self._entries_by_type.clear()
        self._entries_by_name.clear()
        log.debug("Cleared service registry")

    @property
    def registered_types(self) -> frozenset[type]:
        """Get all registered service types.

        Returns
        -------
        frozenset[type]
            Set of registered types.
        """
        return frozenset(self._entries_by_type.keys())

    @property
    def registered_names(self) -> tuple[str, ...]:
        """Get all registered service names.

        Returns
        -------
        tuple[str, ...]
            Tuple of registered names.
        """
        return tuple(sorted(self._entries_by_name.keys()))

    def __len__(self) -> int:
        """Return number of registered services.

        Returns
        -------
        int
            Count of registered services.
        """
        return len(self._entries_by_type)

    def __contains__(self, service_type: type) -> bool:
        """Check if a service type is registered.

        Parameters
        ----------
        service_type
            Type to check.

        Returns
        -------
        bool
            True if registered.
        """
        return service_type in self._entries_by_type


__all__ = [
    "ServiceAlreadyRegisteredError",
    "ServiceEntry",
    "ServiceNotFoundError",
    "ServiceRegistry",
]
