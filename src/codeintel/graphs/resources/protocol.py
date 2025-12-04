"""Resource provider protocol definitions.

This module re-exports unified resource provider types from codeintel.core.resources,
providing backward compatibility for existing graph resource code.

The canonical definitions now live in codeintel.core.resources.
"""

from __future__ import annotations

from typing import TypeVar

from codeintel.core.resources import ResourceProvider, ResourceProviderBase

T = TypeVar("T")

# Re-export core types
__all__ = [
    "BaseResourceProvider",
    "ResourceProvider",
    "ResourceProviderBase",
]


class BaseResourceProvider[T](ResourceProviderBase[T]):
    """Backward-compatible base for graph resource providers.

    This class extends ResourceProviderBase to provide the graph-specific
    interface expected by existing code (using `_create` instead of `_load`).

    Subclasses should implement `_create()` to define resource creation logic.
    """

    def __init__(self, name: str) -> None:
        """Initialize the provider.

        Parameters
        ----------
        name
            Unique resource name.
        """
        super().__init__()
        self._name = name

    @property
    def resource_name(self) -> str:
        """Unique name identifying this resource type.

        Returns
        -------
        str
            Resource type identifier.
        """
        return self._name

    def _load(self) -> T:
        """Load the resource by delegating to _create.

        Returns
        -------
        T
            The created resource.
        """
        return self._create()

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
