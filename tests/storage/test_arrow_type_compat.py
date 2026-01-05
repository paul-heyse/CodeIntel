"""Tests for Arrow type compatibility helpers."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.core.schemas.primitives import Column
from codeintel.core.validation.schema_constraints import is_compatible_arrow_type

pytestmark = pytest.mark.no_runtime_env


def test_decimal_scale_zero_accepts_integer_and_float() -> None:
    """Validate decimal scale=0 compatibility behavior."""
    column = Column(name="amount", type="DECIMAL(10,0)", nullable=False)
    assert is_compatible_arrow_type(column, pa.int64())
    assert is_compatible_arrow_type(column, pa.float64())


def test_decimal_scale_allows_float_when_nonzero() -> None:
    """Validate decimal scale>0 compatibility with floats."""
    column = Column(name="amount", type="DECIMAL(10,2)", nullable=False)
    assert is_compatible_arrow_type(column, pa.float64())


def test_dictionary_encoded_string_is_compatible() -> None:
    """Ensure dictionary-encoded strings are treated as compatible."""
    column = Column(name="status", type="VARCHAR", nullable=False)
    dictionary_type = pa.dictionary(pa.int32(), pa.string())
    assert is_compatible_arrow_type(column, dictionary_type)


def test_nested_types_are_compatible() -> None:
    """Ensure nested LIST/MAP/STRUCT/UNION types are compatible."""
    list_column = Column(name="items", type="LIST(INTEGER)", nullable=True)
    assert is_compatible_arrow_type(list_column, pa.list_(pa.int32()))

    map_column = Column(name="labels", type="MAP(VARCHAR, INTEGER)", nullable=True)
    assert is_compatible_arrow_type(map_column, pa.map_(pa.string(), pa.int32()))

    struct_column = Column(name="payload", type="STRUCT(name VARCHAR)", nullable=True)
    assert is_compatible_arrow_type(struct_column, pa.struct([("name", pa.string())]))

    union_column = Column(name="choice", type="UNION(a INTEGER)", nullable=True)
    union_type = pa.union([pa.field("a", pa.int32())], mode="sparse")
    assert is_compatible_arrow_type(union_column, union_type)
