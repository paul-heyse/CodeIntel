"""Columnar conversion helpers for shared pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import polars as pl
import pyarrow as pa

from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.readers import record_batch_reader_from_batches
from codeintel.core.columnar.stream import (
    ColumnarStream,
    LazyFrameStream,
    coerce_arrow_reader,
    stream_from_reader,
    stream_from_relation,
    stream_from_table,
)
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.duckdb_types import DuckDBRelation

_GOID_COLUMN_MARKER = "goid_h128"
_GOID_COLUMN_TYPE = pl.Decimal(38, 0)

if TYPE_CHECKING:
    type RecordBatchIterable = Iterable[pa.RecordBatch]
    type TabularFrame = pl.LazyFrame
    from codeintel.core.columnar.arrowdsl import ExecutionPlan
    from codeintel.core.columnar.plan_ops import Plan
    type InferableTabularInput = (
        pa.RecordBatchReader
        | pa.Table
        | pl.DataFrame
        | TabularFrame
        | RecordBatchIterable
        | DuckDBRelation
        | Plan
        | ExecutionPlan
    )
else:
    RecordBatchIterable = Iterable[pa.RecordBatch]
    TabularFrame = pl.LazyFrame
    from codeintel.core.columnar.arrowdsl import ExecutionPlan
    from codeintel.core.columnar.plan_ops import Plan
    InferableTabularInput = (
        pa.RecordBatchReader
        | pa.Table
        | pl.DataFrame
        | TabularFrame
        | RecordBatchIterable
        | DuckDBRelation
        | Plan
        | ExecutionPlan
    )


def table_to_reader(
    table: pa.Table,
    *,
    batch_size: int | None = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.RecordBatchReader:
    """Convert an Arrow Table into a RecordBatchReader.

    Returns
    -------
    pa.RecordBatchReader
        Reader over record batches from the table.
    """
    stream = stream_from_table(table, batch_size=batch_size)
    return stream.to_reader(batch_size=batch_size or DEFAULT_ARROW_BATCH_SIZE)


def reader_to_table(reader: pa.RecordBatchReader | pa.Table) -> pa.Table:
    """Materialize a RecordBatchReader into an Arrow Table.

    Returns
    -------
    pa.Table
        Arrow table built from reader batches.
    """
    if isinstance(reader, pa.Table):
        return reader
    return stream_from_reader(reader).to_table()


def lazyframe_to_reader(frame: pl.LazyFrame) -> pa.RecordBatchReader:
    """Convert a Polars LazyFrame into a RecordBatchReader.

    Returns
    -------
    pa.RecordBatchReader
        Reader over streamed record batches.
    """
    stream = _lazyframe_stream(_coerce_goid_columns(frame))
    return stream.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)


def arrow_reader_to_lazyframe(reader: pa.RecordBatchReader) -> pl.LazyFrame:
    """Convert an Arrow RecordBatchReader into a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame constructed from the Arrow reader.

    """
    return _coerce_goid_columns(stream_from_reader(reader).to_lazyframe())


def table_to_lazyframe(table: pa.Table) -> pl.LazyFrame:
    """Convert an Arrow Table into a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame constructed from the Arrow table.
    """
    return _coerce_goid_columns(stream_from_table(table, batch_size=None).to_lazyframe())


def table_to_frame(table: pa.Table) -> pl.DataFrame:
    """Convert an Arrow Table into a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame constructed from the Arrow table.
    """
    return _coerce_goid_columns_frame(table_to_lazyframe(table).collect())


def tabular_to_frame(value: InferableTabularInput) -> pl.DataFrame:
    """Convert an inferable tabular input to a Polars DataFrame.

    Parameters
    ----------
    value
        Tabular input to convert.

    Returns
    -------
    pl.DataFrame
        DataFrame representation of the input.

    """
    if isinstance(value, pl.DataFrame):
        return _coerce_goid_columns_frame(value)
    if isinstance(value, Iterable):
        reader = record_batch_reader_from_iterable(value, empty_policy="none")
        if reader is None:
            return pl.DataFrame()
        return table_to_frame(reader_to_table(reader))
    table = tabular_to_arrow_table(value)
    return table_to_frame(table)


def tabular_to_arrow_reader(
    value: InferableTabularInput,
    *,
    batch_size: int | None = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.RecordBatchReader:
    """Convert an inferable tabular input to a RecordBatchReader.

    Parameters
    ----------
    value
        Tabular input to convert.
    batch_size
        Optional batch size to use when streaming tabular inputs.

    Returns
    -------
    pa.RecordBatchReader
        RecordBatchReader representation of the input.

    Notes
    -----
    RecordBatchReader inputs are single-consume; materialize to a table or
    LazyFrame if reuse is required.
    """
    stream = _coerce_stream(value, batch_size=batch_size)
    return stream.to_reader(batch_size=batch_size or DEFAULT_ARROW_BATCH_SIZE)


def tabular_to_arrow_table(
    value: InferableTabularInput,
    *,
    batch_size: int | None = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.Table:
    """Convert an inferable tabular input to an Arrow Table.

    Parameters
    ----------
    value
        Tabular input to convert.
    batch_size
        Optional batch size to use when streaming tabular inputs.

    Returns
    -------
    pa.Table
        Arrow table representation of the input.

    Notes
    -----
    RecordBatchReader inputs are single-consume; avoid reusing them after
    calling this helper.
    """
    stream = _coerce_stream(value, batch_size=batch_size)
    return stream.to_table()


def reader_from_batches(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for the provided schema and batches.

    Parameters
    ----------
    schema
        Schema to associate with the record batches.
    batches
        Iterable of record batches to stream.

    Returns
    -------
    pa.RecordBatchReader
        Reader yielding the provided record batches.
    """
    return record_batch_reader_from_batches(schema, batches)


def empty_table_from_schema(schema: pa.Schema) -> pa.Table:
    """Return an empty Arrow table with the provided schema.

    Parameters
    ----------
    schema
        Schema for the empty table.

    Returns
    -------
    pa.Table
        Empty Arrow table.
    """
    return pa.Table.from_batches([], schema=schema)


def table_from_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    schema: pa.Schema | None = None,
) -> pa.Table:
    """Return a table built from record batches with optional schema enforcement.

    Parameters
    ----------
    batches
        Iterable of record batches to materialize.
    schema
        Optional schema to enforce on the resulting table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.

    Raises
    ------
    ValueError
        If the batch iterable is empty and no schema is provided.
    """
    reader = record_batch_reader_from_iterable(batches, empty_policy="none")
    if reader is None:
        if schema is None:
            msg = "Record batch iterable is empty; schema must be provided"
            raise ValueError(msg)
        return empty_table_from_schema(schema)
    table = reader_to_table(reader)
    if schema is None or table.schema == schema:
        return table
    return table.cast(schema)


def relation_to_reader(
    relation: DuckDBRelation,
    *,
    batch_size: int | None = DEFAULT_ARROW_BATCH_SIZE,
) -> pa.RecordBatchReader:
    """Return a RecordBatchReader for a DuckDB relation.

    Parameters
    ----------
    relation
        DuckDB relation to stream.
    batch_size
        Optional batch size to use when streaming.

    Returns
    -------
    pa.RecordBatchReader
        Reader yielding relation record batches.
    """
    stream = stream_from_relation(relation, batch_size=batch_size)
    return stream.to_reader(batch_size=batch_size or DEFAULT_ARROW_BATCH_SIZE)


def _record_batch_reader_from_iterable(
    batches: Iterable[pa.RecordBatch],
) -> pa.RecordBatchReader | None:
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        return None
    if not isinstance(first, pa.RecordBatch):
        msg = f"Unsupported tabular input type: {type(first).__name__}"
        raise TypeError(msg)

    def batch_iter() -> Iterable[pa.RecordBatch]:
        yield first
        for batch in iterator:
            if not isinstance(batch, pa.RecordBatch):
                msg = f"Unsupported tabular input type: {type(batch).__name__}"
                raise TypeError(msg)
            yield batch

    return record_batch_reader_from_batches(first.schema, batch_iter())


def _lazyframe_stream(frame: pl.LazyFrame) -> LazyFrameStream:
    return LazyFrameStream(
        frame,
        query_opt_flags=None,
        streaming=True,
        streaming_fallback=True,
        inspect=False,
    )


def _stream_from_native_value(
    value: InferableTabularInput,
    *,
    batch_size: int | None,
) -> ColumnarStream | None:
    stream: ColumnarStream | None = None
    if isinstance(value, (Plan, ExecutionPlan)):
        execution_ctx = resolve_execution_context(None)
        reader = (
            ExecutionPlan.from_plan(value).to_reader(ctx=execution_ctx)
            if isinstance(value, Plan)
            else value.to_reader(ctx=execution_ctx)
        )
        stream = stream_from_reader(reader)
    elif isinstance(value, pa.RecordBatchReader):
        stream = stream_from_reader(value)
    elif isinstance(value, pa.Table):
        stream = stream_from_table(value, batch_size=batch_size)
    elif isinstance(value, DuckDBRelation):
        stream = stream_from_relation(value, batch_size=batch_size)
    elif isinstance(value, pl.LazyFrame):
        stream = _lazyframe_stream(_coerce_goid_columns(value))
    elif isinstance(value, pl.DataFrame):
        stream = stream_from_table(value.to_arrow(), batch_size=batch_size)
    return stream


def _coerce_stream(
    value: InferableTabularInput,
    *,
    batch_size: int | None,
) -> ColumnarStream:
    stream = _stream_from_native_value(value, batch_size=batch_size)
    if stream is None:
        reader = coerce_arrow_reader(value, batch_size=batch_size)
        if reader is not None:
            stream = stream_from_reader(reader)
        elif isinstance(value, Iterable):
            reader = _record_batch_reader_from_iterable(value)
            if reader is None:
                msg = "Unsupported tabular input type: empty iterable"
                raise TypeError(msg)
            stream = stream_from_reader(reader)
    if stream is None:
        msg = f"Unsupported tabular input type: {type(value).__name__}"
        raise TypeError(msg)
    return stream


def record_batch_reader_from_iterable(
    batches: Iterable[pa.RecordBatch],
    *,
    empty_policy: Literal["none", "error"] = "none",
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for an iterable of RecordBatch objects.

    Returns
    -------
    pa.RecordBatchReader | None
        Reader for the batches, or None when empty and allowed by policy.

    Raises
    ------
    ValueError
        If the iterable is empty and the empty policy is set to "error".
    """
    reader = _record_batch_reader_from_iterable(batches)
    if reader is None and empty_policy == "error":
        msg = "Record batch iterable is empty; schema cannot be inferred"
        raise ValueError(msg)
    return reader


def _coerce_goid_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    try:
        columns = frame.collect_schema().names()
    except (AttributeError, ValueError, pl.exceptions.PolarsError):
        return frame
    goid_columns = [
        col for col in columns if isinstance(col, str) and _GOID_COLUMN_MARKER in col.lower()
    ]
    if not goid_columns:
        return frame
    return frame.with_columns(
        [pl.col(name).cast(_GOID_COLUMN_TYPE, strict=False) for name in goid_columns]
    )


def _coerce_goid_columns_frame(frame: pl.DataFrame) -> pl.DataFrame:
    goid_columns = [
        col for col in frame.columns if isinstance(col, str) and _GOID_COLUMN_MARKER in col.lower()
    ]
    if not goid_columns:
        return frame
    return frame.with_columns(
        [pl.col(name).cast(_GOID_COLUMN_TYPE, strict=False) for name in goid_columns]
    )


__all__ = [
    "arrow_reader_to_lazyframe",
    "empty_table_from_schema",
    "lazyframe_to_reader",
    "reader_from_batches",
    "reader_to_table",
    "record_batch_reader_from_iterable",
    "relation_to_reader",
    "table_from_batches",
    "table_to_frame",
    "table_to_lazyframe",
    "table_to_reader",
    "tabular_to_arrow_reader",
    "tabular_to_arrow_table",
    "tabular_to_frame",
]
