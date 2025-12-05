"""Shared protocol for configuration access.

This module defines the ConfigAccessor protocol that provides a common
interface for configuration storage and retrieval, used by both the
lightweight ConfigProvider (for graphs/analytics) and the full-featured
ConfigRegistry (for ingestion).

This protocol enables execution contexts to accept either implementation
while maintaining type safety.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class ConfigAccessor(Protocol):
    """Protocol for typed configuration access.

    Provide a minimal interface for plugins to request configuration
    without needing to know the specific storage implementation.

    This protocol is implemented by:
    - ConfigProvider (core.plugins.context) - Simple dict-based storage
    - ConfigRegistry (core.config_registry) - Full-featured with validation

    Example
    -------
    >>> def process_with_config(accessor: ConfigAccessor) -> None:
    ...     if accessor.has(MyConfig):
    ...         config = accessor.get(MyConfig)
    ...         # use config
    """

    def get[T](self, config_type: type[T]) -> T:
        """Retrieve a required configuration.

        Parameters
        ----------
        config_type
            The configuration type to retrieve.

        Returns
        -------
        T
            The configuration instance.

        Raises
        ------
        KeyError or ValueError
            If the config type is not registered.
        """
        ...

    def get_optional[T](self, config_type: type[T]) -> T | None:
        """Retrieve an optional configuration.

        Parameters
        ----------
        config_type
            The configuration type to retrieve.

        Returns
        -------
        T | None
            The configuration instance or None if not registered.
        """
        ...

    def has(self, config_type: type[object]) -> bool:
        """Check if a configuration type is registered.

        Parameters
        ----------
        config_type
            The type to check.

        Returns
        -------
        bool
            True if registered.
        """
        ...

    def register[T](self, config_type: type[T], config: T) -> None:
        """Register a configuration instance.

        Parameters
        ----------
        config_type
            The type key for registration.
        config
            The configuration instance.
        """
        ...


__all__ = [
    "ConfigAccessor",
]
