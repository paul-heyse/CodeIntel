"""Columnar helpers for CLI streaming output."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping

import pyarrow as pa

from codeintel.cli.core.results import ResultBase
from codeintel.core.columnar.stream import ColumnarStream, RecordBatchReaderStream
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE


def stream_from_items(
    items: Iterable[ResultBase | Mapping[str, object]],
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> ColumnarStream:
    """Return a ColumnarStream for iterable items.

    Parameters
    ----------
    items
        Iterable of ResultBase instances or record mappings.
    batch_size
        Target batch size for chunked record batches.

    Returns
    -------
    ColumnarStream
        Stream adapter wrapping a RecordBatchReader.
    """
    records = (_normalize_item(item) for item in items)
    reader = _record_batch_reader_from_records(records, batch_size=batch_size)
    return RecordBatchReaderStream(reader)


def _normalize_item(item: ResultBase | Mapping[str, object]) -> dict[str, object]:
    if isinstance(item, ResultBase):
        return item.to_dict()
    return dict(item)


def _iter_batches(
    records: Iterable[Mapping[str, object]],
    *,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    iterator = iter(records)
    schema: pa.Schema | None = None
    while True:
        chunk = list(itertools.islice(iterator, batch_size))
        if not chunk:
            break
        if schema is None:
            batch = pa.RecordBatch.from_pylist(chunk)
            schema = batch.schema
        else:
            batch = pa.RecordBatch.from_pylist(chunk, schema=schema)
        yield batch


def _record_batch_reader_from_records(
    records: Iterable[Mapping[str, object]],
    *,
    batch_size: int,
) -> pa.RecordBatchReader:
    batch_iter = _iter_batches(records, batch_size=batch_size)
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        return pa.RecordBatchReader.from_batches(pa.schema([]), [])
    schema = first_batch.schema

    def _all_batches() -> Iterator[pa.RecordBatch]:
        yield first_batch
        yield from batch_iter

    return pa.RecordBatchReader.from_batches(schema, _all_batches())


__all__ = ["stream_from_items"]
