"""Lazy provider decorator.

This module provides a decorator for creating lazy providers
from simple factory functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from codeintel.core.providers.base import LazyProvider

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def lazy_provider(
    name: str | None = None,
) -> Callable[[Callable[[], T]], LazyProvider[T]]:
    """Create a lazy provider from a factory function.

    Parameters
    ----------
    name
        Optional provider name. Defaults to function name.

    Returns
    -------
    Callable[[Callable[[], T]], LazyProvider[T]]
        Decorator that wraps a factory function.

    Examples
    --------
    >>> @lazy_provider("config")
    ... def get_config() -> dict[str, str]:
    ...     return load_config()
    >>> get_config.get()  # First call loads
    {'key': 'value'}
    >>> get_config.is_loaded
    True
    """

    def decorator(factory: Callable[[], T]) -> LazyProvider[T]:
        provider_name = name or factory.__name__
        provider: LazyProvider[T] = LazyProvider(factory, name=provider_name)
        # Copy metadata for introspection (but don't use update_wrapper as it expects callable)
        if hasattr(factory, "__doc__"):
            provider.__doc__ = factory.__doc__
        return provider

    return decorator


def make_provider[T](
    factory: Callable[[], T],
    *,
    name: str | None = None,
) -> LazyProvider[T]:
    """Create a lazy provider from a factory function.

    Parameters
    ----------
    factory
        Callable that produces the resource.
    name
        Optional provider name.

    Returns
    -------
    LazyProvider[T]
        A lazy provider wrapping the factory.

    Examples
    --------
    >>> def load_data() -> list[str]:
    ...     return ["a", "b", "c"]
    >>> provider = make_provider(load_data, name="data")
    >>> provider.get()
    ['a', 'b', 'c']
    """
    provider_name = name or factory.__name__
    return LazyProvider(factory, name=provider_name)


__all__ = [
    "lazy_provider",
    "make_provider",
]
