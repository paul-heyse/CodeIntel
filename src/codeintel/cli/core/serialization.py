"""Generic dataclass serialization utilities.

Provide consistent serialization of result dataclasses to dictionaries
for JSON output.

Note
----
As of v5.0.0, the core serialization infrastructure is defined in
codeintel.core.serialization. This module re-exports key utilities
for backward compatibility.
"""

from __future__ import annotations

# Re-export from core for convenience
from codeintel.core.serialization import (
    SerializableBase,
    SerializableProtocol,
    serialize_value,
)
from codeintel.core.serialization.converters import (
    JsonValue,
    serialize_dataclass_to_dict,
)


def serialize_result(obj: object) -> dict[str, JsonValue]:
    """Serialize any dataclass to dict, handling nested types.

    Parameters
    ----------
    obj
        Dataclass instance to serialize.

    Returns
    -------
    dict[str, JsonValue]
        Dictionary representation of the dataclass.

    Note
    ----
    Raises TypeError if obj is not a dataclass instance.
    """
    return serialize_dataclass_to_dict(obj)


__all__ = [
    "JsonValue",
    "SerializableBase",
    "SerializableProtocol",
    "serialize_result",
    "serialize_value",
]
