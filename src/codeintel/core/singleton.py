"""Thread-safe singleton holder pattern for global registries.

This module provides a reusable pattern for managing global singleton instances
in a thread-safe manner, eliminating the need for `global` statements.

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

from threading import Lock, local
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable


class SingletonNotInitializedError(RuntimeError):
    """Raised when a singleton is requested but not initialized."""

    def __init__(self, cls_name: str) -> None:
        super().__init__(f"{cls_name} singleton not initialized")


class SingletonReentrancyError(RuntimeError):
    """Raised when a singleton is recursively initialized."""

    def __init__(self, cls_name: str) -> None:
        super().__init__(f"{cls_name} singleton initialization is re-entrant")


_THREAD_STATE = local()


def _get_initializing() -> set[type[object]]:
    current = getattr(_THREAD_STATE, "initializing", None)
    if current is None:
        current = set()
        _THREAD_STATE.initializing = current
    return current


class SingletonHolder[T]:
    """Thread-safe singleton holder using double-checked locking.

    Subclass this to create a holder for a specific type. Each subclass
    maintains its own singleton instance.

    Type Parameters
    ---------------
    T
        The type of singleton this holder manages.

    Example
    -------
    >>> class RegistryHolder(SingletonHolder["MyRegistry"]):
    ...     '''Holder for MyRegistry singleton.'''
    ...
    ...     pass
    >>>
    >>> registry = RegistryHolder.get(MyRegistry)
    """

    _instance: ClassVar[object | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init_subclass__(cls) -> None:
        """Initialize per-subclass singleton storage."""
        super().__init_subclass__()
        cls._instance = None
        cls._lock = Lock()

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

        Raises
        ------
        SingletonReentrancyError
            If the singleton factory reenters initialization for the same holder.
        SingletonNotInitializedError
            If the singleton remains uninitialized after invoking the factory.
        """
        if cls._instance is None:
            initializing = _get_initializing()
            if cls in initializing:
                raise SingletonReentrancyError(cls.__name__)
            with cls._lock:
                if cls._instance is None:
                    initializing.add(cls)
                    try:
                        cls._instance = factory()
                    finally:
                        initializing.discard(cls)
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


__all__ = [
    "SingletonHolder",
    "SingletonNotInitializedError",
    "SingletonReentrancyError",
]
