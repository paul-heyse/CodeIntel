"""Tests for tabular step helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pytest

from codeintel.build.hamilton.transforms import tabular_steps

if TYPE_CHECKING:
    from polars import LazyFrame
else:

    class LazyFrame:
        """Typing stub for polars LazyFrame."""


def test_sort_columns_uses_polars_selectors() -> None:
    """Ensure column ordering produces stable output."""
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"b": [1, 2], "a": [3, 4]}).lazy()
    sorted_frame = tabular_steps.sort_columns(frame, ["a", "b"])
    result = cast("LazyFrame", sorted_frame).collect()
    expected = pl.DataFrame({"a": [3, 4], "b": [1, 2]})
    pl.testing.assert_frame_equal(result, expected)


def _reader_for_table(table: pa.Table) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def test_drop_bad_rows_filters_arrow_reader() -> None:
    """Ensure Arrow readers drop rows with nulls in required columns."""
    table = pa.table({"loc": [1, None, 3], "cyclo": [1, 2, None], "name": ["a", "b", "c"]})
    reader = _reader_for_table(table)

    result_reader = cast(
        "pa.RecordBatchReader",
        tabular_steps.drop_bad_rows(reader, ("loc", "cyclo")),
    )
    result = result_reader.read_all().to_pylist()

    assert result == [{"loc": 1, "cyclo": 1, "name": "a"}]


def test_normalize_nulls_drops_arrow_rows() -> None:
    """Ensure Arrow readers drop rows when null policy demands it."""
    table = pa.table({"a": [1, None, 3], "b": ["x", "y", None]})
    reader = _reader_for_table(table)

    result_reader = cast(
        "pa.RecordBatchReader",
        tabular_steps.normalize_nulls(reader, "drop_bad_rows"),
    )
    result = result_reader.read_all().to_pylist()

    assert result == [{"a": 1, "b": "x"}]


def test_clip_numeric_arrow_reader() -> None:
    """Ensure Arrow readers clip numeric columns."""
    table = pa.table({"loc": [1, 20, None], "other": ["a", "b", "c"]})
    reader = _reader_for_table(table)

    result_reader = cast(
        "pa.RecordBatchReader",
        tabular_steps.clip_numeric(reader, "loc", 10),
    )
    result = result_reader.read_all().column("loc").to_pylist()

    assert result == [1, 10, None]


def test_sort_columns_arrow_reader() -> None:
    """Ensure Arrow readers reorder and project columns."""
    table = pa.table({"b": [1, 2], "a": [3, 4], "c": [5, 6]})
    reader = _reader_for_table(table)

    result_reader = cast(
        "pa.RecordBatchReader",
        tabular_steps.sort_columns(reader, ["a", "b"]),
    )
    result_table = result_reader.read_all()

    assert result_table.schema.names == ["a", "b"]
    assert result_table.column("a").to_pylist() == [3, 4]
    assert result_table.column("b").to_pylist() == [1, 2]
