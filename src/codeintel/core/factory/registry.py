"""Factory registry for managing factories.

This module provides a registry for managing factory instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from codeintel.core.factory.protocol import FactoryError

if TYPE_CHECKING:
    from codeintel.core.factory.protocol import FactoryProtocol

log = logging.getLogger(__name__)

T = TypeVar("T")


class FactoryNotFoundError(FactoryError):
    """Raised when a factory is not found in the registry."""

    def __init__(self, factory_name: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        factory_name
            Name of the factory not found.
        """
        super().__init__("registry", f"Factory not found: {factory_name}")
        self.factory_name = factory_name


@dataclass
class FactoryRegistry:
    """Registry for managing factory instances.

    Provides a central location for registering and looking up factories.

    Examples
    --------
    >>> registry = FactoryRegistry()
    >>> registry.register(user_factory)
    >>> factory = registry.get("user")
    """

    _factories: dict[str, FactoryProtocol[Any]] = field(default_factory=dict)

    def register(
        self,
        factory: FactoryProtocol[Any],
        *,
        name: str | None = None,
    ) -> None:
        """Register a factory.

        Parameters
        ----------
        factory
            Factory to register.
        name
            Optional name override.
        """
        factory_name = name or factory.FACTORY_NAME
        self._factories[factory_name] = factory
        log.debug("Registered factory: %s", factory_name)

    def get(self, name: str) -> FactoryProtocol[Any]:
        """Get a factory by name.

        Parameters
        ----------
        name
            Factory name.

        Returns
        -------
        FactoryProtocol[Any]
            The registered factory.

        Raises
        ------
        FactoryNotFoundError
            If factory is not found.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise FactoryNotFoundError(name)
        return factory

    def get_typed[FT](self, name: str, factory_type: type[FT]) -> FT:
        """Get a factory by name with type checking.

        Parameters
        ----------
        name
            Factory name.
        factory_type
            Expected factory type for type inference.

        Returns
        -------
        FT
            The registered factory.

        May raise exceptions from get().
        """
        _ = factory_type  # Used for type inference only
        return cast("FT", self.get(name))

    def has(self, name: str) -> bool:
        """Check if a factory is registered.

        Parameters
        ----------
        name
            Factory name.

        Returns
        -------
        bool
            True if factory is registered.
        """
        return name in self._factories

    def unregister(self, name: str) -> bool:
        """Remove a factory from the registry.

        Parameters
        ----------
        name
            Factory name.

        Returns
        -------
        bool
            True if factory was found and removed.
        """
        if name in self._factories:
            del self._factories[name]
            log.debug("Unregistered factory: %s", name)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered factories."""
        self._factories.clear()
        log.debug("Cleared factory registry")

    @property
    def registered_names(self) -> tuple[str, ...]:
        """Get all registered factory names.

        Returns
        -------
        tuple[str, ...]
            Sorted factory names.
        """
        return tuple(sorted(self._factories.keys()))

    def __len__(self) -> int:
        """Return number of registered factories.

        Returns
        -------
        int
            Count of factories.
        """
        return len(self._factories)

    def __contains__(self, name: str) -> bool:
        """Check if factory is registered.

        Parameters
        ----------
        name
            Factory name.

        Returns
        -------
        bool
            True if registered.
        """
        return name in self._factories


__all__ = [
    "FactoryNotFoundError",
    "FactoryRegistry",
]
