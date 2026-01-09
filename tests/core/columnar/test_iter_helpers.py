"""Streaming-safe iterator tests for columnar helpers."""

from __future__ import annotations

import pyarrow as pa

from codeintel.core.columnar.iter import (
    iter_array_values,
    iter_rows,
    iter_rows_limit,
    iter_tuples,
)


def test_iter_array_values_chunked() -> None:
    """Chunked arrays should yield values without materializing a list."""
    values = pa.chunked_array([[1, None], [2, 3]])
    assert list(iter_array_values(values)) == [1, None, 2, 3]


def test_iter_rows_table_and_batch() -> None:
    """Row iteration should handle tables, batches, and column selection."""
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert list(iter_rows(table)) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert list(iter_rows(table, columns=("b",))) == [{"b": "x"}, {"b": "y"}]
    batch = table.to_batches()[0]
    assert list(iter_rows(batch, columns=("a",))) == [{"a": 1}, {"a": 2}]


def test_iter_tuples_reader() -> None:
    """Tuple iteration should respect column selection for readers."""
    table = pa.table({"x": [1, 2], "y": [3, 4]})
    reader = table.to_reader()
    assert list(iter_tuples(reader, columns=("x",))) == [(1,), (2,)]


def test_iter_rows_limit_reader_across_batches() -> None:
    """Row limiting should stop across batch boundaries."""
    table = pa.table({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    reader = table.to_reader(max_chunksize=2)
    assert list(iter_rows_limit(reader, limit=3)) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
        {"a": 3, "b": "z"},
    ]
