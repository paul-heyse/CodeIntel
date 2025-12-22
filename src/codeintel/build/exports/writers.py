"""Shared export writers for JSONL and Parquet."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, SupportsInt, TextIO, cast, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.protocols import ExportRelation, RecordBatch, RecordBatchReader

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class SupportsIsoformat(Protocol):
    """Protocol for ISO-serializable values."""

    def isoformat(self) -> str:
        """Return the ISO-8601 string representation.

        Returns
        -------
        str
            ISO-8601 formatted string.
        """
        ...


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

    Raises
    ------
    TypeError
        If the object cannot be serialized for JSON output.
    """
    if isinstance(obj, SupportsIsoformat):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        normalized = normalize_decimal_id(obj)
        if normalized is not None:
            return normalized
    msg = f"Type {type(obj)} is not JSON serializable"
    raise TypeError(msg)


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
    """
    rows_written = 0
    reader = rel.fetch_record_batch(batch_size)
    for batch in _iter_batches(reader):
        payload = batch.to_pydict()
        columns = list(payload.keys())
        for idx in range(batch.num_rows):
            record = {name: payload[name][idx] for name in columns}
            if record_type is not None:
                record["_type"] = record_type
            handle.write(json.dumps(record, ensure_ascii=False, default=serializer))
            handle.write("\n")
            rows_written += 1
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


__all__ = [
    "ExportRelation",
    "RecordBatch",
    "RecordBatchReader",
    "SupportsIsoformat",
    "default_json_serializer",
    "write_jsonl_records",
    "write_parquet_relation",
]
