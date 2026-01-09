"""Tests for Arrow-aware cache result storage."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pyarrow as pa
import pytest

from codeintel.build.hamilton.cache_adapter import ArrowCachedResult, ArrowFileResultStore
from codeintel.core.columnar.conversion import reader_to_table, table_to_reader


class _Pylistable(Protocol):
    def to_pylist(self) -> list[object]:
        """Return the table as a list of Python objects."""
        ...


def _reader_from_table(table: pa.Table) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for the provided table.

    Returns
    -------
    pa.RecordBatchReader
        Reader over table batches.
    """
    return table_to_reader(table, batch_size=None)


def _table_from_cached(value: object | None) -> _Pylistable:
    if value is None:
        pytest.fail("Expected cached value but got None")
    if isinstance(value, pa.Table):
        return cast("_Pylistable", value)
    if isinstance(value, pa.RecordBatchReader):
        return reader_to_table(value)
    msg = f"Unexpected cached value type: {type(value)!r}"
    pytest.fail(msg)


def _table_pylist(table: pa.Table) -> list[object]:
    return cast("_Pylistable", table).to_pylist()


def test_arrow_file_result_store_round_trip_reader(tmp_path: Path) -> None:
    """Round-trip RecordBatchReader values through the result store."""
    store = ArrowFileResultStore(path=str(tmp_path))
    table = pa.table({"value": [1, 2, 3]})
    reader = _reader_from_table(table)

    store.set("reader-v1", reader)

    cached = store.get("reader-v1")
    cached_table = _table_from_cached(cached)
    assert cached_table.to_pylist() == _table_pylist(table)


def test_arrow_file_result_store_round_trip_table(tmp_path: Path) -> None:
    """Round-trip Table values through the result store."""
    store = ArrowFileResultStore(path=str(tmp_path))
    table = pa.table({"name": ["a", "b"]})

    store.set("table-v1", table)

    cached = store.get("table-v1")
    assert isinstance(cached, pa.Table)
    assert _table_pylist(cached) == _table_pylist(table)


def test_arrow_file_result_store_round_trip_wrapped_reader(tmp_path: Path) -> None:
    """Round-trip wrapped reader values through the result store."""
    store = ArrowFileResultStore(path=str(tmp_path))
    table = pa.table({"value": [10]})
    wrapped = ArrowCachedResult(kind="reader", table=table)

    store.set("wrapped-v1", wrapped)

    cached = store.get("wrapped-v1")
    cached_table = _table_from_cached(cached)
    assert cached_table.to_pylist() == _table_pylist(table)
