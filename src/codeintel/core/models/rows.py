"""Row model protocols.

This module provides protocols and types for row models
used in database operations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

type RowType = dict[str, Any]


@runtime_checkable
class RowModelProtocol(Protocol):
    """Protocol for row model types.

    Row models provide a consistent interface for converting
    between typed objects and database row dictionaries.

    Examples
    --------
    >>> class UserRow:
    ...     def to_dict(self) -> RowType:
    ...         return {"id": self.id, "name": self.name}
    ...
    ...     @classmethod
    ...     def from_dict(cls, data: RowType) -> "UserRow":
    ...         return cls(data["id"], data["name"])
    """

    def to_dict(self) -> RowType:
        """Convert to a dictionary.

        Returns
        -------
        RowType
            Dictionary representation.
        """
        ...

    @classmethod
    def from_dict(cls, data: RowType) -> RowModelProtocol:
        """Create from a dictionary.

        Parameters
        ----------
        data
            Dictionary data.

        Returns
        -------
        RowModelProtocol
            Model instance.
        """
        ...


@runtime_checkable
class ValidatableRowProtocol(Protocol):
    """Protocol for rows with validation.

    Extends row models with validation support.
    """

    def validate(self) -> list[str]:
        """Validate the row data.

        Returns
        -------
        list[str]
            List of validation errors (empty if valid).
        """
        ...

    @property
    def is_valid(self) -> bool:
        """Check if row is valid.

        Returns
        -------
        bool
            True if valid.
        """
        ...


__all__ = [
    "RowModelProtocol",
    "RowType",
    "ValidatableRowProtocol",
]
