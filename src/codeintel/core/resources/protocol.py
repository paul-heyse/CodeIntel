"""Unified resource provider protocol.

This module defines the core protocol for resource providers that work
with both graph and analytics plugins.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ResourceProvider(Protocol[T_co]):
    """Unified protocol for resource providers.

    Resource providers give plugins access to shared data (graphs, catalogs,
    ASTs, etc.) through a consistent interface. This protocol is used by
    both graph and analytics plugins.

    The provider protocol uses lazy loading - the `get()` method loads and
    returns the resource on demand.

    Type Parameters
    ---------------
    T_co
        The type of resource this provider produces (covariant).

    Attributes
    ----------
    RESOURCE_NAME
        Class variable identifying this resource type for lookup.
    """

    RESOURCE_NAME: ClassVar[str]

    def get(self) -> T_co:
        """Load and return the resource.

        This method may perform lazy loading on first call. Subsequent
        calls should return the cached resource.

        Returns
        -------
        T_co
            The loaded resource.
        """
        ...

    def invalidate(self) -> None:
        """Invalidate the cached resource.

        Providers that support caching should implement this to force
        a reload on the next `get()` call. Providers without caching
        can implement this as a no-op.
        """
        ...


class ResourceProviderBase[T]:
    """Base class for resource providers with common functionality.

    This base class provides a standard implementation pattern for
    resource providers with lazy loading and caching.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.
    """

    RESOURCE_NAME: ClassVar[str] = ""

    def __init__(self) -> None:
        """Initialize the provider."""
        self._cached: T | None = None

    def get(self) -> T:
        """Load and return the resource, caching the result.

        Returns
        -------
        T
            The loaded resource.
        """
        if self._cached is None:
            self._cached = self._load()
        return self._cached

    def _load(self) -> T:
        """Load the resource.

        Subclasses must implement this method to provide the actual
        loading logic.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        msg = f"{self.__class__.__name__} must implement _load()"
        raise NotImplementedError(msg)

    def invalidate(self) -> None:
        """Invalidate the cached resource.

        Call this to force a reload on the next `get()` call.
        """
        self._cached = None


__all__ = [
    "ResourceProvider",
    "ResourceProviderBase",
]
