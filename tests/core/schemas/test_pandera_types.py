"""Tests for shared Pandera dtype mapping."""

from __future__ import annotations

import pandas as pd

from codeintel.core.schemas.pandera_types import dtype_for_column_type


def test_decimal_38_maps_to_object_dtype() -> None:
    """DECIMAL(38,0) should avoid pandas Int64 coercion."""
    assert dtype_for_column_type("DECIMAL(38,0)") is object


def test_decimal_variants_map_to_object_dtype() -> None:
    """Any DECIMAL(...) should map to the 128-bit-safe dtype."""
    assert dtype_for_column_type("DECIMAL(20,0)") is object
    assert dtype_for_column_type("DECIMAL(38,0)") is object


def test_string_maps_to_pandas_string_dtype() -> None:
    """VARCHAR should align with pandas string dtype."""
    expected = pd.StringDtype()
    actual = dtype_for_column_type("VARCHAR")
    assert actual == expected
