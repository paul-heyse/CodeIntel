"""Context builder pattern.

This module provides a generic builder pattern for constructing
execution contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


@dataclass
class ContextBuilder[T]:
    """Generic builder for execution contexts.

    Provides a fluent API for constructing context objects
    with type-safe configuration.

    Type Parameters
    ---------------
    T
        The type of context to build.

    Examples
    --------
    >>> builder: ContextBuilder[MyContext] = ContextBuilder(MyContext)
    >>> ctx = builder.with_value("user_id", "123").with_factory("config", load_config).build()
    """

    _context_type: type[T]
    _values: dict[str, object] = field(default_factory=dict)
    _factories: dict[str, Callable[[], object]] = field(default_factory=dict)

    def with_value(self, key: str, value: object) -> Self:
        """Add a value to the context.

        Parameters
        ----------
        key
            Value key.
        value
            Value to add.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._values[key] = value
        return self

    def with_factory(self, key: str, factory: Callable[[], object]) -> Self:
        """Add a factory for lazy value creation.

        Parameters
        ----------
        key
            Value key.
        factory
            Factory function.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._factories[key] = factory
        return self

    def with_values(self, values: dict[str, object]) -> Self:
        """Add multiple values.

        Parameters
        ----------
        values
            Dictionary of values.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._values.update(values)
        return self

    def build(self) -> T:
        """Build the context.

        Resolves all factories and creates the context instance.

        Returns
        -------
        T
            The constructed context.
        """
        resolved: dict[str, object] = dict(self._values)

        for key, factory in self._factories.items():
            if key not in resolved:
                resolved[key] = factory()

        return self._context_type(**resolved)

    def build_partial(self) -> dict[str, Any]:
        """Build a partial context as a dictionary.

        Useful for constructing contexts that need additional
        values not known at builder time.

        Returns
        -------
        dict[str, Any]
            Resolved values.
        """
        resolved: dict[str, Any] = dict(self._values)

        for key, factory in self._factories.items():
            if key not in resolved:
                resolved[key] = factory()

        return resolved

    @classmethod
    def for_type(cls, context_type: type[T]) -> ContextBuilder[T]:
        """Create a builder for a specific context type.

        Parameters
        ----------
        context_type
            Type of context to build.

        Returns
        -------
        ContextBuilder[T]
            Builder for the context type.
        """
        return cls(_context_type=context_type)


__all__ = [
    "ContextBuilder",
]
