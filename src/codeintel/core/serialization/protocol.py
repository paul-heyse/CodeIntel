"""Serialization protocol definitions.

This module provides the SerializableProtocol for types that can be
serialized to dictionaries and JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping


@runtime_checkable
class SerializableProtocol(Protocol):
    """Protocol for types that can be serialized to dictionaries.

    Implementations provide consistent serialization and deserialization
    for dataclasses and other structured types.

    Examples
    --------
    >>> class MyData:
    ...     def to_dict(self) -> dict[str, object]:
    ...         return {"value": 42}
    ...
    ...     @classmethod
    ...     def from_dict(cls, data: Mapping[str, object]) -> "MyData":
    ...         return cls()
    ...
    ...     def to_json(self, *, indent: int | None = None) -> str:
    ...         return '{"value": 42}'
    """

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the object.
        """
        ...

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Deserialize from dictionary.

        Parameters
        ----------
        data
            Dictionary containing serialized data.

        Returns
        -------
        Self
            Deserialized instance.
        """
        ...

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            Optional indentation for pretty-printing.

        Returns
        -------
        str
            JSON string representation.
        """
        ...


__all__ = [
    "SerializableProtocol",
]
