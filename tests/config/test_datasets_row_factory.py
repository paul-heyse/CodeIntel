"""Tests for row factory utilities in codeintel.core.schemas.row_models."""

from __future__ import annotations

from datetime import datetime
from typing import get_type_hints

import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.row_models import (
    row_model_for_table_schema,
    row_serializer_for_table_schema,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def test_row_model_simple_schema() -> None:
    """Generate row model from a simple schema."""
    schema = TableSchema(
        schema="core",
        name="simple",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
        ],
    )

    result = row_model_for_table_schema(table_schema=schema)

    _require(condition=result.__name__ == "Core__simple__Row", message="name mismatch")
    hints = get_type_hints(result)
    _require(condition=hints["repo"] is str, message="repo type mismatch")
    _require(condition=hints["commit"] is str, message="commit type mismatch")


def test_row_model_nullable_columns_become_optional() -> None:
    """Nullable columns become T | None."""
    schema = TableSchema(
        schema="core",
        name="nullable",
        columns=[
            Column(name="required_col", type="VARCHAR", nullable=False),
            Column(name="optional_col", type="INTEGER", nullable=True),
        ],
    )

    result = row_model_for_table_schema(table_schema=schema)

    hints = get_type_hints(result)
    _require(condition=hints["required_col"] is str, message="required_col type mismatch")

    optional_hint = hints["optional_col"]
    _require(
        condition=hasattr(optional_hint, "__args__"),
        message="optional_col should be Union type",
    )
    _require(condition=int in optional_hint.__args__, message="int should be in union args")
    _require(
        condition=type(None) in optional_hint.__args__,
        message="None should be in union args",
    )


def test_row_model_various_types() -> None:
    """Generate row model with various column types."""
    schema = TableSchema(
        schema="core",
        name="mixed",
        columns=[
            Column(name="string_col", type="VARCHAR", nullable=False),
            Column(name="int_col", type="INTEGER", nullable=False),
            Column(name="float_col", type="DOUBLE", nullable=False),
            Column(name="bool_col", type="BOOLEAN", nullable=False),
            Column(name="timestamp_col", type="TIMESTAMP", nullable=False),
        ],
    )

    result = row_model_for_table_schema(table_schema=schema)

    hints = get_type_hints(result)
    _require(condition=hints["string_col"] is str, message="string_col type mismatch")
    _require(condition=hints["int_col"] is int, message="int_col type mismatch")
    _require(condition=hints["float_col"] is float, message="float_col type mismatch")
    _require(condition=hints["bool_col"] is bool, message="bool_col type mismatch")
    _require(condition=hints["timestamp_col"] is datetime, message="timestamp type mismatch")


def test_row_model_column_order_preserved() -> None:
    """Column order is preserved in generated row model."""
    schema = TableSchema(
        schema="core",
        name="ordered",
        columns=[
            Column(name="z_col", type="VARCHAR", nullable=False),
            Column(name="a_col", type="VARCHAR", nullable=False),
            Column(name="m_col", type="VARCHAR", nullable=False),
        ],
    )

    result = row_model_for_table_schema(table_schema=schema)

    hints = get_type_hints(result)
    keys = list(hints.keys())
    _require(condition=keys == ["z_col", "a_col", "m_col"], message=f"key order mismatch: {keys}")


def test_serializer_simple() -> None:
    """Generate serializer for simple schema."""
    schema = TableSchema(
        schema="core",
        name="simple",
        columns=[
            Column(name="a", type="VARCHAR", nullable=False),
            Column(name="b", type="INTEGER", nullable=False),
        ],
    )

    serialize = row_serializer_for_table_schema(table_schema=schema)
    result = serialize({"a": "hello", "b": 42})

    _require(condition=result == ("hello", 42), message=f"result mismatch: {result}")


def test_serializer_column_order_matches_schema() -> None:
    """Serializer outputs values in schema column order."""
    schema = TableSchema(
        schema="core",
        name="ordered",
        columns=[
            Column(name="z_col", type="VARCHAR", nullable=False),
            Column(name="a_col", type="VARCHAR", nullable=False),
            Column(name="m_col", type="VARCHAR", nullable=False),
        ],
    )

    serialize = row_serializer_for_table_schema(table_schema=schema)
    result = serialize({"a_col": "A", "m_col": "M", "z_col": "Z"})

    _require(condition=result == ("Z", "A", "M"), message=f"result mismatch: {result}")


def test_serializer_handles_none_values() -> None:
    """Serializer handles None values correctly."""
    schema = TableSchema(
        schema="core",
        name="nullable",
        columns=[
            Column(name="required", type="VARCHAR", nullable=False),
            Column(name="optional", type="INTEGER", nullable=True),
        ],
    )

    serialize = row_serializer_for_table_schema(table_schema=schema)
    result = serialize({"required": "test", "optional": None})

    _require(condition=result == ("test", None), message=f"result mismatch: {result}")


def test_serializer_missing_key_raises_error() -> None:
    """Serializer raises KeyError for missing keys."""
    schema = TableSchema(
        schema="core",
        name="simple",
        columns=[
            Column(name="a", type="VARCHAR", nullable=False),
            Column(name="b", type="INTEGER", nullable=False),
        ],
    )

    serialize = row_serializer_for_table_schema(table_schema=schema)

    with pytest.raises(KeyError):
        serialize({"a": "hello"})
