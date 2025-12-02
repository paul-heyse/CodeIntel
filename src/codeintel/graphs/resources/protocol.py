"""Resource provider protocol definitions.

This module defines the base protocol for resource providers, enabling
type-safe dependency injection in graph plugins.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ResourceProvider(Protocol[T_co]):
    """Protocol for resource providers.

    Resource providers supply access to infrastructure resources
    (storage, catalog, engine) in a type-safe, injectable manner.

    Type Parameters
    ---------------
    T_co
        The type of value this provider supplies (covariant).
    """

    @property
    def resource_name(self) -> str:
        """Unique name identifying this resource type.

        Returns
        -------
        str
            Resource type identifier.
        """
        ...

    def get(self) -> T_co:
        """Get the resource value.

        Returns
        -------
        T_co
            The resource instance.
        """
        ...

    def invalidate(self) -> None:
        """Invalidate any cached resource value.

        After invalidation, the next call to get() will create
        a fresh resource instance.
        """
        ...


class BaseResourceProvider[T]:
    """Base implementation of ResourceProvider.

    Provides common functionality for resource providers including
    lazy initialization and cache invalidation.

    Attributes
    ----------
    _name
        Resource name.
    _factory
        Factory function to create the resource.
    _cached
        Cached resource value.
    """

    _name: str
    _cached: T | None

    def __init__(self, name: str) -> None:
        """Initialize the provider.

        Parameters
        ----------
        name
            Unique resource name.
        """
        self._name = name
        self._cached = None

    @property
    def resource_name(self) -> str:
        """Unique name identifying this resource type.

        Returns
        -------
        str
            Resource type identifier.
        """
        return self._name

    def get(self) -> T:
        """Get the resource value, creating if necessary.

        Returns
        -------
        T
            The resource instance.

        Notes
        -----
        May raise `RuntimeError` if resource creation fails.
        """
        if self._cached is None:
            self._cached = self._create()
        return self._cached

    def invalidate(self) -> None:
        """Invalidate the cached resource value."""
        self._cached = None

    def _create(self) -> T:
        """Create a new resource instance.

        Subclasses must implement this method to return the actual resource.

        Raises
        ------
        NotImplementedError
            Always raised in base class.
        """
        message = "Subclasses must implement _create()"
        raise NotImplementedError(message)


__all__ = [
    "BaseResourceProvider",
    "ResourceProvider",
]
