"""Tests for msgspec row struct generation."""

from __future__ import annotations

import msgspec
import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.row_models import (
    row_binding_for_table_schema,
    row_struct_builder_for_table_schema,
    row_struct_for_table_schema,
)


def _sample_schema() -> TableSchema:
    return TableSchema(
        schema="core",
        name="row_struct_demo",
        columns=[
            Column("id", "BIGINT", nullable=False),
            Column("payload", "BLOB"),
            Column("tags", "LIST(VARCHAR)"),
            Column("meta", "STRUCT"),
        ],
        primary_key=("id",),
    )


def test_row_struct_builder_normalizes_payloads() -> None:
    """Row struct builder should normalize payload fields for msgspec."""
    schema = _sample_schema()
    struct_type = row_struct_for_table_schema(table_schema=schema)
    if not issubclass(struct_type, msgspec.Struct):
        pytest.fail("Expected msgspec Struct type for row struct")

    builder = row_struct_builder_for_table_schema(table_schema=schema)
    row = {
        "id": 1,
        "payload": {"example": True},
        "tags": ["alpha", "beta"],
        "meta": {"name": "demo"},
    }
    result = builder(row)
    if not isinstance(result, msgspec.Struct):
        pytest.fail("Expected msgspec Struct instance from builder")
    payload = getattr(result, "payload", None)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        pytest.fail("Expected payload to be encoded as bytes")
    tags = getattr(result, "tags", None)
    if tags != ["alpha", "beta"]:
        pytest.fail("Row struct tags mismatch")


def test_generated_row_binding_includes_struct_helpers() -> None:
    """Generated row bindings should include struct model helpers."""
    schema = _sample_schema()
    binding = row_binding_for_table_schema(table_schema=schema)
    if not issubclass(binding.struct_model, msgspec.Struct):
        pytest.fail("Expected msgspec Struct model on GeneratedRowBinding")
    if not callable(binding.struct_builder):
        pytest.fail("Expected struct_builder on GeneratedRowBinding")
    if not callable(binding.struct_serializer):
        pytest.fail("Expected struct_serializer on GeneratedRowBinding")
