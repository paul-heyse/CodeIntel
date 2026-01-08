"""Columnar row buffering helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.conversion import record_batch_reader_from_iterable
from codeintel.core.columnar.readers import empty_reader_from_schema
from codeintel.core.columnar.schema_alignment import align_table_to_contract
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.schemas.arrow_gen import (
    ArrowSchemaMetadata,
    ExtrasPolicy,
    arrow_contract_for_table_schema,
)
from codeintel.core.schemas.row_models import normalize_row_value_for_type
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import ColumnType

ColumnarRows = dict[str, list[object]]


@dataclass(slots=True)
class ColumnarRowBuffer:
    """Mutable buffer for building columnar row payloads."""

    table_key: str
    columns: tuple[str, ...]
    column_types: tuple[ColumnType, ...]
    data: ColumnarRows
    row_count: int = 0

    def append(self, row: Mapping[str, object]) -> None:
        """Append a row mapping to the buffer."""
        for name, col_type in zip(self.columns, self.column_types, strict=True):
            self.data[name].append(normalize_row_value_for_type(row[name], col_type))
        self.row_count += 1

    def extend(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Append multiple rows to the buffer."""
        for row in rows:
            self.append(row)


@dataclass(slots=True)
class ColumnarBatchCollector:
    """Buffer rows into Arrow RecordBatches with a fixed batch size."""

    table_key: str
    columns: tuple[str, ...]
    column_types: tuple[ColumnType, ...]
    arrow_schema: pa.Schema
    batch_size: int
    batches: list[pa.RecordBatch] = field(default_factory=list)
    row_count: int = 0
    _buffer: ColumnarRowBuffer | None = None

    def append(self, row: Mapping[str, object]) -> None:
        """Append a row, flushing to a RecordBatch when the batch is full."""
        if self._buffer is None:
            self._buffer = _buffer_from_columns(
                table_key=self.table_key,
                columns=self.columns,
                column_types=self.column_types,
            )
        self._buffer.append(row)
        self.row_count += 1
        if self._buffer.row_count >= self.batch_size:
            self._flush()

    def extend(self, rows: Iterable[Mapping[str, object]]) -> None:
        """Append multiple rows to the collector."""
        for row in rows:
            self.append(row)

    def append_buffer(self, buffer: ColumnarRowBuffer) -> None:
        """Append a columnar buffer as a RecordBatch.

        Parameters
        ----------
        buffer
            Buffer containing columnar row data to append.
        """
        if buffer.row_count == 0:
            return
        self._flush()
        batch = pa.RecordBatch.from_pydict(buffer.data, schema=self.arrow_schema)
        self.batches.append(batch)
        self.row_count += buffer.row_count

    def flush(self) -> None:
        """Flush any buffered rows into a RecordBatch."""
        self._flush()

    def to_table(self) -> pa.Table:
        """Return a materialized Arrow table for the collected batches.

        Returns
        -------
        pa.Table
            Table built from the collected RecordBatches.
        """
        self._flush()
        if not self.batches:
            return pa.Table.from_batches([], schema=self.arrow_schema)
        return pa.Table.from_batches(self.batches, schema=self.arrow_schema)

    def to_reader(self) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for the collected batches.

        Returns
        -------
        pa.RecordBatchReader
            Reader over the collected RecordBatches.
        """
        self._flush()
        reader = record_batch_reader_from_iterable(self.batches, empty_policy="none")
        if reader is None:
            return empty_reader_from_schema(self.arrow_schema)
        return reader

    def _flush(self) -> None:
        if self._buffer is None or self._buffer.row_count == 0:
            return
        batch = pa.RecordBatch.from_pydict(self._buffer.data, schema=self.arrow_schema)
        self.batches.append(batch)
        self._buffer = None


def columnar_buffer_for_table_key(table_key: str) -> ColumnarRowBuffer:
    """Create a ColumnarRowBuffer using the table schema registry.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    ColumnarRowBuffer
        Buffer seeded with table columns and types.
    """
    schema = get_schema_service().require_table_schema(table_key)
    columns = tuple(schema.column_names())
    column_types: tuple[ColumnType, ...] = tuple(column.type for column in schema.columns)
    return ColumnarRowBuffer(
        table_key=table_key,
        columns=columns,
        column_types=column_types,
        data={name: [] for name in columns},
    )


def empty_table_for_table(table_key: str) -> pa.Table:
    """Return an empty Arrow table using the table schema.

    Returns
    -------
    pa.Table
        Empty table configured with the table schema.
    """
    arrow_schema = _arrow_schema_for_table(table_key, extras_policy=None)
    return pa.Table.from_batches([], schema=arrow_schema)


def columnar_batch_collector_for_table_key(
    table_key: str,
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
    extras_policy: ExtrasPolicy | None = None,
) -> ColumnarBatchCollector:
    """Create a ColumnarBatchCollector seeded from the table schema.

    Returns
    -------
    ColumnarBatchCollector
        Collector configured with schema columns and the requested batch size.
    """
    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(table_key)
    columns = tuple(table_schema.column_names())
    column_types: tuple[ColumnType, ...] = tuple(column.type for column in table_schema.columns)
    arrow_schema = _arrow_schema_for_table(table_key, extras_policy=extras_policy)
    return ColumnarBatchCollector(
        table_key=table_key,
        columns=columns,
        column_types=column_types,
        arrow_schema=arrow_schema,
        batch_size=batch_size,
    )


def table_for_rows(
    table_key: str,
    rows: Iterable[Mapping[str, object]] | Iterable[Sequence[object]],
    *,
    batch_size: int = DEFAULT_ARROW_BATCH_SIZE,
    extras_policy: ExtrasPolicy | None = None,
) -> tuple[pa.Table, int]:
    """Build a table from row mappings using the contract schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row mappings or row sequences aligned to the table schema columns.
    batch_size
        Target batch size for buffering row data.
    extras_policy
        Optional extras policy to apply when aligning to the contract schema.

    Returns
    -------
    tuple[pa.Table, int]
        Table for the row batches plus the total row count.
    """
    collector = columnar_batch_collector_for_table_key(
        table_key,
        batch_size=batch_size,
        extras_policy=extras_policy,
    )
    collector.extend(_iter_row_mappings(table_key, rows))
    if collector.row_count == 0:
        return empty_table_for_table(table_key), 0
    return collector.to_table(), collector.row_count


def _iter_row_mappings(
    table_key: str,
    rows: Iterable[Mapping[str, object]] | Iterable[Sequence[object]],
) -> Iterable[Mapping[str, object]]:
    rows_iter = iter(rows)
    try:
        first = next(rows_iter)
    except StopIteration:
        return iter(())
    if isinstance(first, Mapping):
        return _iter_mapping_rows(first, rows_iter)
    if _is_row_sequence(first):
        columns = tuple(get_schema_service().require_table_schema(table_key).column_names())
        return _iter_sequence_rows(first, rows_iter, columns)
    msg = f"Unsupported row payload for {table_key}: {type(first)}"
    raise TypeError(msg)


def _iter_mapping_rows(
    first: Mapping[str, object],
    rows_iter: Iterable[Mapping[str, object] | Sequence[object]],
) -> Iterable[Mapping[str, object]]:
    yield first
    for row in rows_iter:
        if not isinstance(row, Mapping):
            msg = f"Mixed row payloads in mapping stream: {type(row)}"
            raise TypeError(msg)
        yield row


def _iter_sequence_rows(
    first: Sequence[object],
    rows_iter: Iterable[Mapping[str, object] | Sequence[object]],
    columns: Sequence[str],
) -> Iterable[Mapping[str, object]]:
    yield dict(zip(columns, first, strict=True))
    for row in rows_iter:
        if not _is_row_sequence(row):
            msg = f"Mixed row payloads in sequence stream: {type(row)}"
            raise TypeError(msg)
        yield dict(zip(columns, row, strict=True))


def _is_row_sequence(row: object) -> bool:
    return isinstance(row, Sequence) and not isinstance(row, (bytes, str))


def table_for_columnar_rows(
    table_key: str,
    rows: Mapping[str, Sequence[object]],
    *,
    extras_policy: ExtrasPolicy | None = None,
) -> tuple[pa.Table, int]:
    """Build a table from columnar row data using the contract schema.

    Returns
    -------
    tuple[pa.Table, int]
        Table for the row batches plus the total row count.
    """
    row_count = columnar_row_count(rows)
    if row_count == 0:
        arrow_schema = _arrow_schema_for_table(table_key, extras_policy=extras_policy)
        return pa.Table.from_batches([], schema=arrow_schema), 0
    contract_schema = _arrow_schema_for_table(table_key, extras_policy=extras_policy)
    normalized = {name: list(values) for name, values in rows.items()}
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for name, values in normalized.items():
        if name in contract_schema.names:
            field = contract_schema.field(name)
            arrays.append(pa.array(values, type=field.type))
            fields.append(field)
            continue
        array = pa.array(values)
        arrays.append(array)
        fields.append(pa.field(name, array.type))
    batch = pa.record_batch(arrays, schema=pa.schema(fields))
    table = pa.Table.from_batches([batch], schema=batch.schema)
    aligned = align_table_to_contract(table, contract_schema, extras_policy=extras_policy)
    return aligned, row_count


def _arrow_schema_for_table(
    table_key: str,
    extras_policy: ExtrasPolicy | None,
) -> pa.Schema:
    schema_service = get_schema_service()
    if extras_policy is None:
        arrow_schema = schema_service.get_arrow_schema(table_key)
        if arrow_schema is not None:
            return arrow_schema
    table_schema = schema_service.require_table_schema(table_key)
    metadata = None if extras_policy is None else ArrowSchemaMetadata(extras_policy=extras_policy)
    return arrow_contract_for_table_schema(table_schema=table_schema, metadata=metadata)


def _buffer_from_columns(
    *,
    table_key: str,
    columns: tuple[str, ...],
    column_types: tuple[ColumnType, ...],
) -> ColumnarRowBuffer:
    return ColumnarRowBuffer(
        table_key=table_key,
        columns=columns,
        column_types=column_types,
        data={name: [] for name in columns},
    )


def columnar_row_count(columns: Mapping[str, Sequence[object]]) -> int:
    """Return row count for a columnar mapping, validating lengths.

    Parameters
    ----------
    columns
        Columnar mapping of column names to sequences of values.

    Returns
    -------
    int
        Number of rows represented by the columnar mapping.

    Raises
    ------
    ValueError
        If the columnar mapping contains columns with mismatched lengths.
    """
    lengths = {len(values) for values in columns.values()}
    if not lengths:
        return 0
    if len(lengths) > 1:
        msg = f"Column lengths mismatch: {sorted(lengths)}"
        raise ValueError(msg)
    return lengths.pop()


__all__ = [
    "ColumnarBatchCollector",
    "ColumnarRowBuffer",
    "ColumnarRows",
    "columnar_batch_collector_for_table_key",
    "columnar_buffer_for_table_key",
    "columnar_row_count",
    "empty_table_for_table",
    "table_for_columnar_rows",
    "table_for_rows",
]
