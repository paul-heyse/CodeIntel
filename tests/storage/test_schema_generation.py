"""Tests for TypedDict JSON Schema generation helpers."""

from __future__ import annotations

import typing
from typing import TypedDict

from codeintel.storage.schema_generation import json_schema_from_typeddict, validate_row_with_schema


class SampleRow(TypedDict):
    """Minimal row model for schema generation tests."""

    name: str
    count: int
    flag: bool


def test_json_schema_from_typeddict_round_trip() -> None:
    """
    Generated schema should validate a conforming row.

    Raises
    ------
    AssertionError
        When schema shape differs from expectations.
    """
    schema = json_schema_from_typeddict(SampleRow)
    if schema["type"] != "object":
        message = "Expected object schema"
        raise AssertionError(message)
    required_raw = typing.cast("list[object]", schema.get("required", []))
    required = {str(key) for key in required_raw}
    if required != {"name", "count", "flag"}:
        message = "Required keys mismatch"
        raise AssertionError(message)
    validate_row_with_schema({"name": "x", "count": 1, "flag": True}, schema)
