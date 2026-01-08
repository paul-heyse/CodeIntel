"""RecordBatchReader helpers for Arrow columnar workflows."""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa


def record_batch_reader_from_batches(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader from an iterable of record batches.

    Parameters
    ----------
    schema
        Schema to associate with the record batches.
    batches
        Iterable of Arrow record batches.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader over the provided record batches.
    """
    return pa.RecordBatchReader.from_batches(schema, batches)


def empty_reader_from_schema(schema: pa.Schema) -> pa.RecordBatchReader:
    """Return an empty reader with the provided schema.

    Parameters
    ----------
    schema
        Schema to associate with the empty reader.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader with no batches and the provided schema.
    """
    return record_batch_reader_from_batches(schema, [])


__all__ = ["empty_reader_from_schema", "record_batch_reader_from_batches"]
