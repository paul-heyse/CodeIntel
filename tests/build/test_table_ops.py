"""Tests for table operation helpers."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.table_ops import drop_table_columns
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_drop_table_columns_preserves_row_count_when_empty() -> None:
    """drop_table_columns should preserve row count when all columns are removed."""
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})

    dropped = drop_table_columns(table, ["a", "b"])

    expect_equal(dropped.column_names, [])
    expect_equal(dropped.num_rows, 3)
