"""Unified serialization types for CodeIntel.

This module provides the canonical serialization protocol and utilities
used across all modules for consistent to_dict/from_dict patterns.

Examples
--------
>>> from codeintel.core.serialization import SerializableProtocol, serialize_value
>>> from dataclasses import dataclass
>>> @dataclass
... class MyData:
...     name: str
...     count: int
"""

from __future__ import annotations

from codeintel.core.serialization.base import SerializableBase, serialize_dataclass
from codeintel.core.serialization.converters import (
    deserialize_value,
    serialize_value,
)
from codeintel.core.serialization.protocol import SerializableProtocol

__all__ = [
    "SerializableBase",
    "SerializableProtocol",
    "deserialize_value",
    "serialize_dataclass",
    "serialize_value",
]
