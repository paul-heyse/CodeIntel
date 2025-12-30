"""Shared export writers for JSONL and Parquet."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, SupportsInt, TextIO, cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.core.exports.codecs import coerce_export_value, encode_batch
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.protocols import ExportRelation, RecordBatch, RecordBatchReader

if TYPE_CHECKING:
    from pathlib import Path


def default_json_serializer(obj: object) -> object:
    """Serialize objects for JSON output.

    Parameters
    ----------
    obj
        Object to serialize.

    Returns
    -------
    object
        JSON-serializable representation.
    """
    return coerce_export_value(obj)


def _coerce_row_count(value: object) -> int:
    """Coerce a row count value to int with basic validation.

    Returns
    -------
    int
        Normalized integer row count.

    Raises
    ------
    TypeError
        If the value is a boolean.
    ValueError
        If the value cannot be converted to a valid integer.
    """
    if isinstance(value, bool):
        msg = f"Invalid row count type: {type(value).__name__}"
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        msg = f"Invalid row count value: {value}"
        raise ValueError(msg)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            msg = f"Invalid row count value: {value}"
            raise ValueError(msg) from exc
    try:
        return int(cast("SupportsInt", value))
    except (TypeError, ValueError) as exc:
        msg = f"Invalid row count value: {value}"
        raise ValueError(msg) from exc


def write_jsonl_records(
    handle: TextIO,
    *,
    rel: ExportRelation,
    record_type: str | None = None,
    serializer: Callable[[object], object] = default_json_serializer,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> int:
    """Write JSONL records from a DuckDB relation.

    Parameters
    ----------
    handle
        File-like object opened for writing.
    rel
        DuckDB relation to stream records from.
    record_type
        Optional record type field to inject.
    serializer
        Callable used to serialize values.
    batch_size
        Maximum rows per batch read from DuckDB.

    Returns
    -------
    int
        Number of rows written to the JSONL output.

    Raises
    ------
    ValueError
        If a custom serializer is provided.
    """
    if serializer is not default_json_serializer:
        msg = "Custom JSON serializers are not supported for columnar JSONL exports"
        raise ValueError(msg)
    reader = rel.fetch_record_batch(batch_size)
    return write_jsonl_reader(handle, reader=reader, record_type=record_type)


def write_jsonl_reader(
    handle: TextIO,
    *,
    reader: RecordBatchReader,
    record_type: str | None = None,
) -> int:
    """Write JSONL records from a RecordBatchReader.

    Parameters
    ----------
    handle
        File-like object opened for writing.
    reader
        Arrow record batch reader to stream.
    record_type
        Optional record type field to inject.

    Returns
    -------
    int
        Number of rows written to the JSONL output.
    """
    rows_written = 0
    if record_type is None:
        for batch in _iter_batches(reader):
            if batch.num_rows == 0:
                continue
            rows_written += batch.num_rows
            for chunk in encode_batch(batch, schema=reader.schema):
                handle.write(chunk.decode("utf-8"))
        return rows_written
    for batch in _iter_batches(reader):
        frame = _frame_for_batch(batch)
        if record_type is not None:
            frame = frame.with_columns(pl.lit(record_type).alias("_type"))
        if frame.height == 0:
            continue
        frame.write_ndjson(handle)
        rows_written += frame.height
    return rows_written


def write_json_array(
    handle: TextIO,
    *,
    reader: RecordBatchReader,
    record_type: str | None = None,
) -> int:
    """Write a JSON array to the handle from a RecordBatchReader.

    Parameters
    ----------
    handle
        File-like object opened for writing.
    reader
        Arrow record batch reader to stream.
    record_type
        Optional record type field to inject.

    Returns
    -------
    int
        Number of rows written to the JSON output.
    """
    rows_written = 0
    first = True
    handle.write("[")
    for batch in _iter_batches(reader):
        frame = _frame_for_batch(batch)
        if record_type is not None:
            frame = frame.with_columns(pl.lit(record_type).alias("_type"))
        if frame.height == 0:
            continue
        payload = frame.write_json()
        if payload == "[]":
            continue
        content = payload[1:-1]
        if not content:
            continue
        if first:
            first = False
        else:
            handle.write(",")
        handle.write(content)
        rows_written += frame.height
    handle.write("]\n")
    return rows_written


def write_parquet_relation(
    *,
    rel: ExportRelation,
    output_path: Path,
    batch_size: int = 10_000,
) -> int:
    """Write a DuckDB relation to Parquet and return row count.

    Parameters
    ----------
    rel
        DuckDB relation to export.
    output_path
        Destination path for the Parquet file.
    batch_size
        Maximum rows per batch read from DuckDB.

    Returns
    -------
    int
        Number of rows written to the Parquet file.
    """
    write_parquet = getattr(rel, "write_parquet", None)
    if write_parquet is not None:
        write_parquet(str(output_path))
        row_count_row = rel.aggregate("count(*)").fetchone()
        return _coerce_row_count(row_count_row[0]) if row_count_row else 0

    reader = rel.fetch_record_batch(batch_size)
    rows_written = 0
    wrote_batches = False
    with pq.ParquetWriter(str(output_path), reader.schema) as writer:
        for batch in _iter_batches(reader):
            rows_written += batch.num_rows
            wrote_batches = True
            writer.write_table(pa.Table.from_batches([batch], schema=reader.schema))
    if not wrote_batches:
        pq.write_table(pa.Table.from_batches([], schema=reader.schema), str(output_path))
    return rows_written


def _iter_batches(reader: Iterable[RecordBatch]) -> Iterable[RecordBatch]:
    """Yield record batches from a batch reader.

    Parameters
    ----------
    reader
        Iterable of record batches.

    Returns
    -------
    Iterable[RecordBatch]
        Record batches to process.
    """
    return reader


def _frame_for_batch(batch: RecordBatch) -> pl.DataFrame:
    frame = pl.from_arrow(batch)
    if isinstance(frame, pl.Series):
        return frame.to_frame()
    return frame


__all__ = [
    "ExportRelation",
    "RecordBatch",
    "RecordBatchReader",
    "default_json_serializer",
    "write_json_array",
    "write_jsonl_reader",
    "write_jsonl_records",
    "write_parquet_relation",
]
