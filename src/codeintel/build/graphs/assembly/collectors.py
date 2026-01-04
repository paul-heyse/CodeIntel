"""Row/collector helpers for Arrow-first graph assembly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from codeintel.core.columnar.rows import (
    ColumnarBatchCollector,
    columnar_batch_collector_for_table_key,
    empty_reader_for_table,
    record_batch_reader_for_columnar_rows,
    record_batch_reader_for_rows,
)
from codeintel.core.schemas.arrow_gen import ExtrasPolicy


def collector_for_table(
    table_key: str,
    *,
    batch_size: int | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> ColumnarBatchCollector:
    """Create a batch collector for the requested table.

    Returns
    -------
    ColumnarBatchCollector
        Collector configured with the table schema.
    """
    if batch_size is None:
        return columnar_batch_collector_for_table_key(table_key, extras_policy=extras_policy)
    return columnar_batch_collector_for_table_key(
        table_key,
        batch_size=batch_size,
        extras_policy=extras_policy,
    )


def reader_for_rows(
    table_key: str,
    rows: Iterable[Mapping[str, object]],
    *,
    batch_size: int | None = None,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.RecordBatchReader:
    """Build a reader from row mappings using the table contract.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the table contract.
    """
    if batch_size is None:
        reader, _ = record_batch_reader_for_rows(table_key, rows, extras_policy=extras_policy)
        return reader
    reader, _ = record_batch_reader_for_rows(
        table_key,
        rows,
        batch_size=batch_size,
        extras_policy=extras_policy,
    )
    return reader


def reader_for_columnar_rows(
    table_key: str,
    rows: Mapping[str, Sequence[object]],
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> pa.RecordBatchReader:
    """Build a reader from columnar row data using the table contract.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the table contract.
    """
    reader, _ = record_batch_reader_for_columnar_rows(
        table_key,
        rows,
        extras_policy=extras_policy,
    )
    return reader


def empty_reader(table_key: str) -> pa.RecordBatchReader:
    """Return an empty RecordBatchReader for the table.

    Returns
    -------
    pyarrow.RecordBatchReader
        Empty reader configured with the table schema.
    """
    return empty_reader_for_table(table_key)


__all__ = [
    "ColumnarBatchCollector",
    "collector_for_table",
    "empty_reader",
    "reader_for_columnar_rows",
    "reader_for_rows",
]
