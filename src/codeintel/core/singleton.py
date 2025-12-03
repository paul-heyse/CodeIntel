"""Thread-safe singleton holder pattern for global registries.

This module provides a reusable pattern for managing global singleton instances
in a thread-safe manner, eliminating the need for `global` statements and
their associated `# noqa: PLW0603` suppressions.

Example
-------
>>> from codeintel.core.singleton import SingletonHolder
>>>
>>> class MyRegistryHolder(SingletonHolder["MyRegistry"]):
...     '''Holder for MyRegistry singleton.'''
...
...     pass
>>>
>>> def get_registry() -> MyRegistry:
...     return MyRegistryHolder.get(MyRegistry)
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class SingletonNotInitializedError(RuntimeError):
    """Raised when a singleton is requested but not initialized."""

    def __init__(self, cls_name: str) -> None:
        super().__init__(f"{cls_name} singleton not initialized")


class SingletonHolder(Generic[T]):
    """Thread-safe singleton holder using double-checked locking.

    Subclass this to create a holder for a specific type. Each subclass
    maintains its own singleton instance.

    Type Parameters
    ---------------
    T
        The type of singleton this holder manages.

    Example
    -------
    >>> class ConfigRegistryHolder(SingletonHolder[ConfigRegistry]):
    ...     '''Holder for ConfigRegistry singleton.'''
    ...
    ...     pass
    >>>
    >>> registry = ConfigRegistryHolder.get(ConfigRegistry)
    """

    _instance: ClassVar[object | None] = None
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def get(cls, factory: Callable[[], T]) -> T:
        """Return the singleton instance, creating it if necessary.

        Uses double-checked locking for thread safety with minimal
        lock contention after initialization.

        Parameters
        ----------
        factory
            Callable that creates a new instance if one doesn't exist.

        Returns
        -------
        T
            The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = factory()
        if cls._instance is None:
            raise SingletonNotInitializedError(cls.__name__)
        return cast("T", cls._instance)

    @classmethod
    def get_or_none(cls) -> T | None:
        """Return the singleton instance if it exists, None otherwise.

        Returns
        -------
        T | None
            The singleton instance or None if not initialized.
        """
        return cast("T | None", cls._instance)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance to None.

        Primarily useful for testing to ensure clean state between tests.
        Thread-safe.
        """
        with cls._lock:
            cls._instance = None

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the singleton has been initialized.

        Returns
        -------
        bool
            True if the singleton instance exists.
        """
        return cls._instance is not None


__all__ = ["SingletonHolder"]
