"""Tests for JSON Schema validator factory and registry access."""

from __future__ import annotations

import pytest

from codeintel.storage.schema_generation import build_validator, validate_row_with_schema


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
