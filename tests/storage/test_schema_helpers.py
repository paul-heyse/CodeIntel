"""Tests for schema generation, validation, and table creation helpers."""

from __future__ import annotations

import typing
from typing import TypedDict

import pytest

from codeintel.storage.schema.json_schema import (
    build_validator,
    json_schema_from_typeddict,
    validate_row_with_schema,
)

if typing.TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# =============================================================================
# Test Data
# =============================================================================


class SampleRow(TypedDict):
    """Minimal row model for schema generation tests."""

    name: str
    count: int
    flag: bool


# =============================================================================
# JSON Schema Generation Tests
# =============================================================================


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


# =============================================================================
# Validator Factory Tests
# =============================================================================


def test_build_validator_accepts_mapping() -> None:
    """Validator factory should accept mapping schema and expose schema attribute."""
    schema: dict[str, object] = {"type": "object", "properties": {"a": {"type": "string"}}}
    validator = build_validator(schema)
    if getattr(validator, "schema", None) != dict(schema):
        pytest.fail("Validator should retain provided schema mapping")


def test_validate_row_with_schema_passes_valid_data() -> None:
    """Row validation should pass for conforming data."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    validate_row_with_schema({"a": "ok"}, schema)


# =============================================================================
# Schema Table Creation Tests
# =============================================================================


def test_apply_all_schemas_creates_function_validation(schema_gateway: StorageGateway) -> None:
    """Schema application should create analytics.function_validation."""
    con = schema_gateway.con
    rows = con.execute("PRAGMA table_info(analytics.function_validation)").fetchall()
    if not rows:
        pytest.fail("analytics.function_validation should exist after apply_all_schemas")
