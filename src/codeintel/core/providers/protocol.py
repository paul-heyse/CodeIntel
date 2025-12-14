"""Provider protocol definitions.

This module defines the core protocols for resource providers,
providing a standardized interface for lazy resource loading.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class ProviderProtocol[T](Protocol):
    """Protocol for resource providers.

    Providers give access to shared resources through a consistent
    interface with lazy loading support.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.

    Attributes
    ----------
    PROVIDER_NAME
        Class variable identifying this provider type.

    Examples
    --------
    >>> class ConfigProvider:
    ...     PROVIDER_NAME: ClassVar[str] = "config"
    ...
    ...     def get(self) -> dict[str, str]:
    ...         return self._load_config()
    ...
    ...     def refresh(self) -> None:
    ...         self._cached = None
    ...
    ...     @property
    ...     def is_loaded(self) -> bool:
    ...         return self._cached is not None
    """

    PROVIDER_NAME: ClassVar[str]

    def get(self) -> T:
        """Get the provided resource.

        Load the resource if not already loaded, then return it.

        Returns
        -------
        T
            The loaded resource.
        """
        ...

    def refresh(self) -> None:
        """Force refresh of cached resource.

        Clear any cached state and reload on next get() call.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Check whether resource is currently loaded.

        Returns
        -------
        bool
            True if the resource is loaded and cached.
        """
        ...


@runtime_checkable
class OptionalProviderProtocol[T](Protocol):
    """Protocol for providers that may not have data.

    Extends the basic provider with optional access that
    returns None instead of raising exceptions.

    Type Parameters
    ---------------
    T
        The type of resource this provider produces.
    """

    def get(self) -> T:
        """Get the provided resource.

        Returns
        -------
        T
            The loaded resource.

        Raises
        ------
        ProviderError
            If the resource cannot be loaded.
        """
        ...

    def get_or_none(self) -> T | None:
        """Get the resource or None if unavailable.

        Returns
        -------
        T | None
            The loaded resource, or None if unavailable.
        """
        ...

    def refresh(self) -> None:
        """Force refresh of cached resource."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check whether resource is currently loaded.

        Returns
        -------
        bool
            True if the resource is loaded.
        """
        ...


class ProviderError(Exception):
    """Base exception for provider-related errors.

    Attributes
    ----------
    provider_name
        Name of the provider that raised the error.
    """

    def __init__(self, provider_name: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        provider_name
            Name of the provider.
        message
            Error message.
        """
        super().__init__(f"[{provider_name}] {message}")
        self.provider_name = provider_name


class ProviderNotLoadedError(ProviderError):
    """Raised when a provider cannot load its resource."""


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider's resource is not available."""


__all__ = [
    "OptionalProviderProtocol",
    "ProviderError",
    "ProviderNotAvailableError",
    "ProviderNotLoadedError",
    "ProviderProtocol",
]
