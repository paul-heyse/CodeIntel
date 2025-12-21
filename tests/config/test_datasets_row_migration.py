"""Tests for schema-generated row bindings."""

from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.row_models import row_binding_for_table_schema

_SHA256_HEX_LEN: int = 64


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def test_row_binding_includes_provenance() -> None:
    """Row bindings include provenance metadata for the source schema."""
    schema = TableSchema(
        schema="test",
        name="example",
        columns=[
            Column(name="id", type="INTEGER", nullable=False),
        ],
    )

    binding = row_binding_for_table_schema(table_schema=schema)

    _require(condition=binding.table_key == "test.example", message="table_key mismatch")
    _require(
        condition=len(binding.schema_hash) == _SHA256_HEX_LEN,
        message="schema_hash length mismatch",
    )
    _require(condition=is_dataclass(binding.row_model), message="row_model should be a dataclass")


def test_row_binding_serializer_preserves_order() -> None:
    """Row binding serializer preserves schema column order."""
    schema = TableSchema(
        schema="test",
        name="ordering",
        columns=[
            Column(name="first", type="VARCHAR", nullable=False),
            Column(name="second", type="INTEGER", nullable=False),
        ],
    )

    binding = row_binding_for_table_schema(table_schema=schema)
    row = {"second": 2, "first": "a"}

    _require(
        condition=binding.serializer(row) == ("a", 2),
        message="serializer did not preserve column order",
    )
