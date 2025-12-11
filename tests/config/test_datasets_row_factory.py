"""Tests for codeintel.config.datasets.row_factory module."""

from __future__ import annotations

from typing import get_type_hints

import pandas as pd
import pytest
from pandera import Column, DataFrameSchema

from codeintel.config.datasets.row_factory import (
    row_serializer_from_pandera,
    typed_dict_from_pandera,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


# ------------------------------------------------------------------
# typed_dict_from_pandera tests
# ------------------------------------------------------------------


def test_typeddict_simple_schema() -> None:
    """Generate TypedDict from simple schema."""
    schema = DataFrameSchema(
        {
            "repo": Column(str),
            "commit": Column(str),
        }
    )

    result = typed_dict_from_pandera("SimpleRow", schema)

    _require(condition=result.__name__ == "SimpleRow", message="name mismatch")
    hints = get_type_hints(result)
    _require(condition=hints["repo"] is str, message="repo type mismatch")
    _require(condition=hints["commit"] is str, message="commit type mismatch")


def test_typeddict_nullable_columns_become_optional() -> None:
    """Nullable columns become T | None when nullable_as_optional=True."""
    schema = DataFrameSchema(
        {
            "required_col": Column(str, nullable=False),
            "optional_col": Column(int, nullable=True),
        }
    )

    result = typed_dict_from_pandera("NullableRow", schema)

    hints = get_type_hints(result)
    _require(condition=hints["required_col"] is str, message="required_col type mismatch")
    # Check that optional_col allows None
    # The type will be int | None which is a UnionType
    optional_hint = hints["optional_col"]
    _require(
        condition=hasattr(optional_hint, "__args__"), message="optional_col should be Union type"
    )
    _require(condition=int in optional_hint.__args__, message="int should be in union args")
    _require(condition=type(None) in optional_hint.__args__, message="None should be in union args")


def test_typeddict_nullable_as_optional_false() -> None:
    """Nullable columns remain base type when nullable_as_optional=False."""
    schema = DataFrameSchema(
        {
            "optional_col": Column(int, nullable=True),
        }
    )

    result = typed_dict_from_pandera("NonOptionalRow", schema, nullable_as_optional=False)

    hints = get_type_hints(result)
    _require(condition=hints["optional_col"] is int, message="optional_col type mismatch")


def test_typeddict_various_dtypes() -> None:
    """Generate TypedDict with various dtypes."""
    schema = DataFrameSchema(
        {
            "string_col": Column(str),
            "int_col": Column(pd.Int64Dtype()),
            "float_col": Column(pd.Float64Dtype()),
            "bool_col": Column(pd.BooleanDtype()),
        }
    )

    result = typed_dict_from_pandera("MixedRow", schema)

    hints = get_type_hints(result)
    _require(condition=hints["string_col"] is str, message="string_col type mismatch")
    _require(condition=hints["int_col"] is int, message="int_col type mismatch")
    _require(condition=hints["float_col"] is float, message="float_col type mismatch")
    _require(condition=hints["bool_col"] is bool, message="bool_col type mismatch")


def test_typeddict_column_order_preserved() -> None:
    """Column order is preserved in generated TypedDict."""
    schema = DataFrameSchema(
        {
            "z_col": Column(str),
            "a_col": Column(str),
            "m_col": Column(str),
        }
    )

    result = typed_dict_from_pandera("OrderedRow", schema)

    # TypedDict annotations maintain insertion order
    hints = get_type_hints(result)
    keys = list(hints.keys())
    _require(condition=keys == ["z_col", "a_col", "m_col"], message=f"key order mismatch: {keys}")


# ------------------------------------------------------------------
# row_serializer_from_pandera tests
# ------------------------------------------------------------------


def test_serializer_simple() -> None:
    """Generate serializer for simple schema."""
    schema = DataFrameSchema(
        {
            "a": Column(str),
            "b": Column(int),
        }
    )

    serialize = row_serializer_from_pandera(schema)
    result = serialize({"a": "hello", "b": 42})

    _require(condition=result == ("hello", 42), message=f"result mismatch: {result}")


def test_serializer_column_order_matches_schema() -> None:
    """Serializer outputs values in schema column order."""
    schema = DataFrameSchema(
        {
            "z_col": Column(str),
            "a_col": Column(str),
            "m_col": Column(str),
        }
    )

    serialize = row_serializer_from_pandera(schema)
    result = serialize({"a_col": "A", "m_col": "M", "z_col": "Z"})

    # Order should match schema: z_col, a_col, m_col
    _require(condition=result == ("Z", "A", "M"), message=f"result mismatch: {result}")


def test_serializer_handles_none_values() -> None:
    """Serializer handles None values correctly."""
    schema = DataFrameSchema(
        {
            "required": Column(str),
            "optional": Column(int, nullable=True),
        }
    )

    serialize = row_serializer_from_pandera(schema)
    result = serialize({"required": "test", "optional": None})

    _require(condition=result == ("test", None), message=f"result mismatch: {result}")


def test_serializer_missing_key_raises_error() -> None:
    """Serializer raises KeyError for missing keys."""
    schema = DataFrameSchema(
        {
            "a": Column(str),
            "b": Column(int),
        }
    )

    serialize = row_serializer_from_pandera(schema)

    with pytest.raises(KeyError):
        serialize({"a": "hello"})  # Missing "b"


# ------------------------------------------------------------------
# _pandera_dtype_to_python tests (via integration)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "expected_type"),
    [
        (pd.Int64Dtype(), int),
        (pd.Float64Dtype(), float),
        (pd.BooleanDtype(), bool),
        (pd.StringDtype(), str),
    ],
)
def test_dtype_mapping_pandas_types(dtype: object, expected_type: type[object]) -> None:
    """Map pandas extension dtypes to Python types."""
    schema = DataFrameSchema({"col": Column(dtype)})  # type: ignore[arg-type]
    result = typed_dict_from_pandera("TestRow", schema)
    hints = get_type_hints(result)
    _require(
        condition=hints["col"] is expected_type,
        message=f"dtype {dtype} should map to {expected_type}",
    )
