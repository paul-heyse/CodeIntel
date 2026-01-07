"""Arrow iteration helpers for row-wise fallbacks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pyarrow as pa


def iter_array_values(values: pa.Array | pa.ChunkedArray) -> Iterator[object]:
    """Yield Python values without materializing a list.

    Yields
    ------
    object
        Python scalar values.
    """
    if isinstance(values, pa.ChunkedArray):
        for chunk in values.iterchunks():
            for item in chunk:
                yield item.as_py()
        return
    for item in values:
        yield item.as_py()


def iter_rows(
    table_or_batch: pa.Table | pa.RecordBatch,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, object]]:
    """Yield row dicts without `to_pylist()` materialization.

    Yields
    ------
    dict[str, object]
        Row dictionaries.
    """
    if isinstance(table_or_batch, pa.Table):
        column_names = list(columns) if columns is not None else list(table_or_batch.column_names)
        if not column_names:
            return
        selected = table_or_batch.select(column_names)
        for batch in selected.to_batches():
            yield from iter_rows(batch, column_names)
        return
    batch = table_or_batch
    column_names = list(columns) if columns is not None else list(batch.schema.names)
    if not column_names:
        return
    arrays = [batch.column(column_name) for column_name in column_names]
    for row_index in range(batch.num_rows):
        yield {
            column_name: arrays[idx][row_index].as_py()
            for idx, column_name in enumerate(column_names)
        }


def iter_batches(
    table_or_reader: pa.Table | pa.RecordBatchReader,
    *,
    batch_size: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Yield record batches from a table or reader.

    Parameters
    ----------
    table_or_reader
        Table or record batch reader to iterate.
    batch_size
        Optional max chunk size for table batch iteration.

    Yields
    ------
    pyarrow.RecordBatch
        Record batches from the source.
    """
    if isinstance(table_or_reader, pa.RecordBatchReader):
        yield from table_or_reader
        return
    if batch_size is None:
        yield from table_or_reader.to_batches()
    else:
        yield from table_or_reader.to_batches(max_chunksize=batch_size)


__all__ = [
    "iter_array_values",
    "iter_batches",
    "iter_rows",
]
