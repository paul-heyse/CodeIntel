"""Deprecated wrapper for ingestion frame helpers.

These helpers now return Arrow readers to keep ingestion Arrow-first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from codeintel.core.columnar.rows import (
    columnar_row_count,
    empty_reader_for_table,
    record_batch_reader_for_columnar_rows,
)

ColumnsSpec = Mapping[str, Sequence[object]] | Sequence[str] | None


def _normalize_columns(columns: ColumnsSpec) -> Mapping[str, Sequence[object]]:
    if columns is None:
        return {}
    if isinstance(columns, Mapping):
        return columns
    return {str(name): [] for name in columns}


def dedupe_frame_for_table(_table_key: str, frame: pa.Table) -> pa.Table:
    """Return the input frame unchanged for Arrow-first ingestion.

    Returns
    -------
    pyarrow.Table
        Dedupe is handled upstream; this is a passthrough for compatibility.
    """
    return frame


def lazyframe_for_ingest_columns(
    table_key: str, rows: Mapping[str, Sequence[object]]
) -> pa.RecordBatchReader:
    """Return an Arrow reader for ingest columnar rows.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the provided columnar rows.
    """
    reader, _ = record_batch_reader_for_columnar_rows(table_key, rows)
    return reader


def lazyframe_for_table_columns(
    table_key: str,
    columns: ColumnsSpec,
) -> pa.RecordBatchReader:
    """Return an Arrow reader aligned to the table schema.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the table schema for the requested columns.
    """
    normalized = _normalize_columns(columns)
    if not normalized or columnar_row_count(normalized) == 0:
        try:
            return empty_reader_for_table(table_key)
        except (KeyError, RuntimeError):
            return pa.RecordBatchReader.from_batches(pa.schema([]), [])
    reader, _ = record_batch_reader_for_columnar_rows(table_key, normalized)
    return reader


def empty_lazyframe_for_table(table_key: str) -> pa.RecordBatchReader:
    """Return an empty Arrow reader using the table schema.

    Returns
    -------
    pyarrow.RecordBatchReader
        Empty reader aligned to the table schema.
    """
    return empty_reader_for_table(table_key)


__all__ = [
    "ColumnsSpec",
    "dedupe_frame_for_table",
    "empty_lazyframe_for_table",
    "lazyframe_for_ingest_columns",
    "lazyframe_for_table_columns",
]
