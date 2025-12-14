"""Factory protocol definitions.

This module defines the core protocols for factory implementations,
providing a standardized interface for creating objects.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class FactoryProtocol[T](Protocol):
    """Protocol for factory implementations.

    Factories provide a consistent interface for creating objects,
    abstracting away the construction details.

    Type Parameters
    ---------------
    T
        The type of object this factory creates.

    Attributes
    ----------
    FACTORY_NAME
        Unique identifier for this factory type.

    Examples
    --------
    >>> class UserFactory:
    ...     FACTORY_NAME: ClassVar[str] = "user"
    ...
    ...     def create(self, **kwargs: object) -> User:
    ...         return User(**kwargs)
    ...
    ...     def can_create(self, **kwargs: object) -> bool:
    ...         return "name" in kwargs
    """

    FACTORY_NAME: ClassVar[str]

    def create(self, **kwargs: object) -> T:
        """Create an instance of the target type.

        Parameters
        ----------
        **kwargs
            Arguments for creating the instance.

        Returns
        -------
        T
            The created instance.
        """
        ...

    def can_create(self, **kwargs: object) -> bool:
        """Check if the factory can create with the given arguments.

        Parameters
        ----------
        **kwargs
            Arguments to check.

        Returns
        -------
        bool
            True if creation would succeed.
        """
        ...


@runtime_checkable
class CachingFactoryProtocol[T](Protocol):
    """Protocol for factories that cache created instances.

    Type Parameters
    ---------------
    T
        The type of object this factory creates.
    """

    def get_or_create(self, key: str, **kwargs: object) -> T:
        """Get a cached instance or create a new one.

        Parameters
        ----------
        key
            Cache key for the instance.
        **kwargs
            Arguments for creating the instance if not cached.

        Returns
        -------
        T
            The cached or newly created instance.
        """
        ...

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached instance.

        Parameters
        ----------
        key
            Cache key to invalidate.

        Returns
        -------
        bool
            True if the key was found and invalidated.
        """
        ...


class FactoryError(Exception):
    """Base exception for factory-related errors.

    Attributes
    ----------
    factory_name
        Name of the factory that raised the error.
    """

    def __init__(self, factory_name: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        factory_name
            Name of the factory.
        message
            Error message.
        """
        super().__init__(f"[{factory_name}] {message}")
        self.factory_name = factory_name


class FactoryCreationError(FactoryError):
    """Raised when factory creation fails."""


__all__ = [
    "CachingFactoryProtocol",
    "FactoryCreationError",
    "FactoryError",
    "FactoryProtocol",
]
