"""Base context implementation.

This module provides a base class for execution contexts with
common patterns for resource access and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import MutableMapping


@dataclass
class BaseContext:
    """Base class for execution contexts.

    Provides a minimal context implementation with metadata
    and extensibility support.

    Attributes
    ----------
    context_id
        Unique identifier for this context.
    metadata
        Additional context metadata.

    Examples
    --------
    >>> ctx = BaseContext(context_id="run-123")
    >>> ctx.set_meta("user", "admin")
    >>> ctx.get_meta("user")
    'admin'
    """

    context_id: str | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def get_meta[U](self, key: str, default: U | None = None) -> U | None:
        """Get metadata value.

        Parameters
        ----------
        key
            Metadata key.
        default
            Default value if not found.

        Returns
        -------
        U | None
            Metadata value or default.
        """
        return cast("U | None", self.metadata.get(key, default))

    def set_meta(self, key: str, value: object) -> None:
        """Set metadata value.

        Parameters
        ----------
        key
            Metadata key.
        value
            Value to set.
        """
        self.metadata[key] = value

    def has_meta(self, key: str) -> bool:
        """Check if metadata key exists.

        Parameters
        ----------
        key
            Metadata key.

        Returns
        -------
        bool
            True if key exists.
        """
        return key in self.metadata

    def clear_meta(self) -> None:
        """Clear all metadata."""
        self.metadata.clear()


__all__ = [
    "BaseContext",
]
