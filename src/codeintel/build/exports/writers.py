"""Shared export writers for JSONL and Parquet."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, SupportsInt, TextIO, TypedDict, cast

import msgspec
import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.exports.serialization import coerce_export_row, coerce_export_value
from codeintel.storage.protocols import ExportRelation, RecordBatch, RecordBatchReader

if TYPE_CHECKING:
    from pathlib import Path


class _ParquetWriterKwargs(TypedDict, total=False):
    use_dictionary: bool


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
    encoder = msgspec.json.Encoder()
    for batch in _iter_batches(reader):
        rows = _json_rows_from_batch(batch, record_type=record_type)
        if not rows:
            continue
        payload = encoder.encode_lines(rows)
        handle.write(payload.decode("utf-8"))
        rows_written += len(rows)
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
    encoder = msgspec.json.Encoder()
    handle.write("[")
    for batch in _iter_batches(reader):
        rows = _json_rows_from_batch(batch, record_type=record_type)
        for row in rows:
            payload = encoder.encode(row).decode("utf-8")
            if first:
                first = False
            else:
                handle.write(",")
            handle.write(payload)
        rows_written += len(rows)
    handle.write("]\n")
    return rows_written


def write_parquet_relation(
    *,
    rel: ExportRelation,
    output_path: Path,
    batch_size: int = 10_000,
    dictionary_encode: bool = False,
    dictionary_columns: Sequence[str] | None = None,
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
    dictionary_encode
        Whether to dictionary encode all columns during export.
    dictionary_columns
        Optional explicit columns to dictionary encode.

    Returns
    -------
    int
        Number of rows written to the Parquet file.
    """
    write_parquet = getattr(rel, "write_parquet", None)
    if write_parquet is not None and not (dictionary_encode or dictionary_columns):
        write_parquet(str(output_path))
        row_count_row = rel.aggregate("count(*)").fetchone()
        return _coerce_row_count(row_count_row[0]) if row_count_row else 0

    reader = rel.fetch_record_batch(batch_size)
    return write_parquet_reader(
        reader=reader,
        output_path=output_path,
        dictionary_encode=dictionary_encode,
        dictionary_columns=dictionary_columns,
    )


def write_parquet_reader(
    *,
    reader: RecordBatchReader,
    output_path: Path,
    dictionary_encode: bool = False,
    dictionary_columns: Sequence[str] | None = None,
) -> int:
    """Write a RecordBatchReader to Parquet and return row count.

    Parameters
    ----------
    reader
        Arrow record batch reader to export.
    output_path
        Destination path for the Parquet file.
    dictionary_encode
        Whether to dictionary encode all columns during export.
    dictionary_columns
        Optional explicit columns to dictionary encode.

    Returns
    -------
    int
        Number of rows written to the Parquet file.
    """
    rows_written = 0
    wrote_batches = False
    writer_kwargs = _parquet_writer_kwargs(
        dictionary_encode=dictionary_encode,
        dictionary_columns=dictionary_columns,
    )
    with pq.ParquetWriter(str(output_path), reader.schema, **writer_kwargs) as writer:
        for batch in _iter_batches(reader):
            rows_written += batch.num_rows
            wrote_batches = True
            table = pa.Table.from_batches([cast("pa.RecordBatch", batch)], schema=reader.schema)
            writer.write_table(table)
    if not wrote_batches:
        pq.write_table(
            pa.Table.from_batches([], schema=reader.schema),
            str(output_path),
            **writer_kwargs,
        )
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


def _parquet_writer_kwargs(
    *,
    dictionary_encode: bool,
    dictionary_columns: Sequence[str] | None,
) -> _ParquetWriterKwargs:
    if dictionary_columns:
        return {"use_dictionary": True}
    if dictionary_encode:
        return {"use_dictionary": True}
    return {}


def _json_rows_from_batch(
    batch: RecordBatch,
    *,
    record_type: str | None,
) -> list[dict[str, object]]:
    table = pa.Table.from_batches([cast("pa.RecordBatch", batch)], schema=batch.schema)
    rows: list[dict[str, object]] = table.to_pylist()
    if record_type is not None:
        for row in rows:
            row["_type"] = record_type
    return [coerce_export_row(row) for row in rows]


__all__ = [
    "ExportRelation",
    "RecordBatch",
    "RecordBatchReader",
    "default_json_serializer",
    "write_json_array",
    "write_jsonl_reader",
    "write_jsonl_records",
    "write_parquet_reader",
    "write_parquet_relation",
]
