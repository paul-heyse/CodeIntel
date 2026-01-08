"""Shared export writers for JSONL and Parquet."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import SupportsInt, TypedDict, cast

import msgspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from codeintel.build.tabular.arrow_ops import json_writer_available, write_json_streaming
from codeintel.build.tabular.conversion import record_batch_reader_from_iterable
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.exports.arrow_ipc import default_ipc_write_options, iter_ipc_stream
from codeintel.core.exports.serialization import coerce_export_row, coerce_export_value
from codeintel.core.ports.export import ExportRelation, RecordBatch, RecordBatchReader


class _ParquetWriterKwargs(TypedDict, total=False):
    use_dictionary: bool


class _CountingBatchIterator:
    def __init__(self, batches: Iterable[RecordBatch]) -> None:
        self._iterator = iter(batches)
        self.rows = 0

    def __iter__(self) -> _CountingBatchIterator:
        return self

    def __next__(self) -> RecordBatch:
        batch = next(self._iterator)
        self.rows += batch.num_rows
        return batch


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
    output_path: Path,
    *,
    rel: ExportRelation,
    record_type: str | None = None,
    serializer: Callable[[object], object] = default_json_serializer,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
) -> int:
    """Write JSONL records from a DuckDB relation.

    Parameters
    ----------
    output_path
        Output path for JSONL records.
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
    return write_jsonl_reader(output_path, reader=reader, record_type=record_type)


def write_jsonl_reader(
    output_path: Path,
    *,
    reader: RecordBatchReader,
    record_type: str | None = None,
) -> int:
    """Write JSONL records from a RecordBatchReader.

    Parameters
    ----------
    output_path
        Output path for JSONL records.
    reader
        Arrow record batch reader to stream.
    record_type
        Optional record type field to inject.

    Returns
    -------
    int
        Number of rows written to the JSONL output.
    """
    if record_type is None and json_writer_available():
        counting_iter = _CountingBatchIterator(_iter_batches(reader))
        writer_reader = record_batch_reader_from_iterable(counting_iter, empty_policy="none")
        if writer_reader is None:
            empty_reader = pa.RecordBatchReader.from_batches(reader.schema, [])
            write_json_streaming(empty_reader, output_path)
            return 0
        write_json_streaming(writer_reader, output_path)
        return counting_iter.rows

    rows_written = 0
    encoder = msgspec.json.Encoder()
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in _iter_batches(reader):
            if batch.num_rows == 0:
                continue
            rows = _iter_json_rows_from_batch(batch, record_type=record_type)
            payload = encoder.encode_lines(rows)
            handle.write(payload.decode("utf-8"))
            rows_written += batch.num_rows
    return rows_written


def write_json_array(
    output_path: Path,
    *,
    reader: RecordBatchReader,
    record_type: str | None = None,
) -> int:
    """Write a JSON array to the handle from a RecordBatchReader.

    Parameters
    ----------
    output_path
        Output path for JSON array output.
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
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for batch in _iter_batches(reader):
            for row in _iter_json_rows_from_batch(batch, record_type=record_type):
                payload = encoder.encode(row).decode("utf-8")
                if first:
                    first = False
                else:
                    handle.write(",")
                handle.write(payload)
            rows_written += batch.num_rows
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
            table = _maybe_dictionary_encode_table(table, dictionary_columns)
            writer.write_table(table)
    if not wrote_batches:
        empty_table = pa.Table.from_batches([], schema=reader.schema)
        empty_table = _maybe_dictionary_encode_table(empty_table, dictionary_columns)
        pq.write_table(
            empty_table,
            str(output_path),
            **writer_kwargs,
        )
    return rows_written


def write_arrow_reader(
    output_path: Path,
    *,
    reader: RecordBatchReader,
    metadata: Mapping[str, object] | None = None,
    batch_metadata: Mapping[str, object] | None = None,
) -> int:
    """Write a RecordBatchReader to an Arrow IPC stream and return row count.

    Parameters
    ----------
    output_path
        Destination path for the Arrow IPC stream.
    reader
        Arrow record batch reader to export.
    metadata
        Optional schema metadata to attach to the IPC stream.
    batch_metadata
        Optional per-batch metadata to attach to record batches.

    Returns
    -------
    int
        Number of rows written to the IPC stream.
    """
    counting_iter = _CountingBatchIterator(_iter_batches(reader))
    writer_reader = record_batch_reader_from_iterable(counting_iter, empty_policy="none")
    if writer_reader is None:
        empty_reader = pa.RecordBatchReader.from_batches(reader.schema, [])
        _write_arrow_stream(
            output_path,
            reader=empty_reader,
            metadata=metadata,
            batch_metadata=batch_metadata,
        )
        return 0
    _write_arrow_stream(
        output_path,
        reader=writer_reader,
        metadata=metadata,
        batch_metadata=batch_metadata,
    )
    return counting_iter.rows


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


def _maybe_dictionary_encode_table(
    table: pa.Table,
    dictionary_columns: Sequence[str] | None,
) -> pa.Table:
    if not dictionary_columns:
        return table
    encode = getattr(pc, "dictionary_encode", None)
    if not callable(encode):
        return table
    encode_set = set(dictionary_columns)
    arrays: list[pa.Array | pa.ChunkedArray] = []
    fields: list[pa.Field] = []
    for name in table.schema.names:
        column = table.column(name)
        field = table.schema.field(name)
        encoded: pa.Array | pa.ChunkedArray | None = None
        if name in encode_set and (
            pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
        ):
            with contextlib.suppress(
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
                ValueError,
            ):
                candidate = encode(column)
                if isinstance(candidate, (pa.Array, pa.ChunkedArray)):
                    encoded = candidate
        if encoded is None:
            arrays.append(column)
            fields.append(field)
        else:
            arrays.append(encoded)
            fields.append(field.with_type(encoded.type))
    if not arrays:
        return table
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata))


def _write_arrow_stream(
    output_path: Path,
    *,
    reader: RecordBatchReader,
    metadata: Mapping[str, object] | None,
    batch_metadata: Mapping[str, object] | None,
) -> None:
    options = default_ipc_write_options()
    with output_path.open("wb") as handle:
        for chunk in iter_ipc_stream(
            reader,
            metadata=metadata,
            batch_metadata=batch_metadata,
            options=options,
        ):
            handle.write(chunk)


def _iter_json_rows_from_batch(
    batch: RecordBatch,
    *,
    record_type: str | None,
) -> Iterable[dict[str, object]]:
    record_batch = cast("pa.RecordBatch", batch)
    columns = record_batch.to_pydict()
    if not columns:
        return
    names = list(columns.keys())
    values_iter = zip(*(columns[name] for name in names), strict=False)
    for values in values_iter:
        row = dict(zip(names, values, strict=False))
        if record_type is not None:
            row["_type"] = record_type
        yield coerce_export_row(row)


__all__ = [
    "ExportRelation",
    "RecordBatch",
    "RecordBatchReader",
    "default_json_serializer",
    "write_arrow_reader",
    "write_json_array",
    "write_jsonl_reader",
    "write_jsonl_records",
    "write_parquet_reader",
    "write_parquet_relation",
]
