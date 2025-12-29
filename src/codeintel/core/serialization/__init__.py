"""Serialization utilities for CodeIntel.

This module provides the canonical value conversion helpers used across
the codebase for JSON-compatible payloads.
"""

from __future__ import annotations

from codeintel.core.serialization.converters import (
    deserialize_value,
    serialize_dataclass_to_dict,
    serialize_value,
)
from codeintel.core.serialization.stable import JsonValue, stable_json_value, stable_stringify

__all__ = [
    "JsonValue",
    "deserialize_value",
    "serialize_dataclass_to_dict",
    "serialize_value",
    "stable_json_value",
    "stable_stringify",
]
