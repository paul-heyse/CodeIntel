"""Base serialization mixin.

This module provides SerializableBase, a mixin class that adds
standard serialization methods to dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Self, get_type_hints

from codeintel.core.serialization.converters import (
    deserialize_value,
    serialize_dataclass_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class SerializableBase:
    """Mixin providing standard serialization for dataclasses.

    Add this as a base class to dataclasses to get automatic
    to_dict(), from_dict(), and to_json() methods.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class MyData(SerializableBase):
    ...     name: str
    ...     count: int
    >>> data = MyData(name="test", count=42)
    >>> data.to_dict()
    {'name': 'test', 'count': 42}
    >>> MyData.from_dict({"name": "test", "count": 42})
    MyData(name='test', count=42)
    """

    def to_dict(self, *, omit_none: bool = False) -> dict[str, object]:
        """Serialize to dictionary.

        Parameters
        ----------
        omit_none
            If True, omit fields with None values.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result = serialize_dataclass_to_dict(self, omit_none=omit_none)
        # Cast is safe: JsonValue is a subset of object
        return dict(result)

    @classmethod
    def from_dict(cls: type[Self], data: Mapping[str, object]) -> Self:
        """Deserialize from dictionary.

        Parameters
        ----------
        data
            Dictionary containing field values.

        Returns
        -------
        Self
            New instance with values from dictionary.

        Raises
        ------
        TypeError
            If class is not a dataclass.
        """
        if not is_dataclass(cls):
            msg = f"{cls.__name__} must be a dataclass to use from_dict"
            raise TypeError(msg)

        # Get type hints for deserialization
        try:
            hints = get_type_hints(cls)
        except (NameError, AttributeError, TypeError):
            # NameError: Forward reference not resolvable
            # AttributeError: Class doesn't support get_type_hints
            # TypeError: Invalid annotation
            hints = {}

        # Build kwargs from dataclass fields
        kwargs: dict[str, object] = {}
        for field in fields(cls):
            if field.name in data:
                raw_value = data[field.name]
                target_type = hints.get(field.name)
                # Convert to JsonValue for deserialize_value
                if isinstance(raw_value, (str, int, float, bool, dict, list, type(None))):
                    kwargs[field.name] = deserialize_value(raw_value, target_type)
                else:
                    kwargs[field.name] = raw_value

        return cls(**kwargs)

    def to_json(self, *, indent: int | None = None, omit_none: bool = False) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        indent
            Optional indentation for pretty-printing.
        omit_none
            If True, omit fields with None values.

        Returns
        -------
        str
            JSON string representation.
        """
        return json.dumps(self.to_dict(omit_none=omit_none), indent=indent)


def serialize_dataclass(obj: object, *, omit_none: bool = False) -> dict[str, object]:
    """Serialize a dataclass to dictionary.

    This is a convenience function that delegates to serialize_dataclass_to_dict.

    Parameters
    ----------
    obj
        Dataclass instance to serialize.
    omit_none
        If True, omit fields with None values.

    Returns
    -------
    dict[str, object]
        Dictionary representation.
    """
    result = serialize_dataclass_to_dict(obj, omit_none=omit_none)
    return dict(result)


__all__ = [
    "SerializableBase",
    "serialize_dataclass",
]
