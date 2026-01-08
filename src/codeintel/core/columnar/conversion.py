"""Columnar conversion helpers for shared pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.duckdb_types import DuckDBRelation

_GOID_COLUMN_MARKER = "goid_h128"
_GOID_COLUMN_TYPE = pl.Decimal(38, 0)

if TYPE_CHECKING:
    type RecordBatchIterable = Iterable[pa.RecordBatch]
    type TabularFrame = pl.LazyFrame
    type InferableTabularInput = (
        pa.RecordBatchReader
        | pa.Table
        | pl.DataFrame
        | TabularFrame
        | RecordBatchIterable
        | DuckDBRelation
    )
else:
    RecordBatchIterable = Iterable[pa.RecordBatch]
    TabularFrame = pl.LazyFrame
    InferableTabularInput = (
        pa.RecordBatchReader
        | pa.Table
        | pl.DataFrame
        | TabularFrame
        | RecordBatchIterable
        | DuckDBRelation
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
    to_reader = getattr(table, "to_reader", None)
    if callable(to_reader):
        if batch_size is None:
            return to_reader()
        try:
            return to_reader(max_chunksize=batch_size)
        except TypeError:
            return to_reader()
    batches = table.to_batches(max_chunksize=batch_size) if batch_size else table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


def reader_to_table(reader: pa.RecordBatchReader | pa.Table) -> pa.Table:
    """Materialize a RecordBatchReader into an Arrow Table.

    Returns
    -------
    pa.Table
        Arrow table built from reader batches.
    """
    if isinstance(reader, pa.Table):
        return reader
    return pa.Table.from_batches(reader, schema=reader.schema)


def lazyframe_to_reader(frame: pl.LazyFrame) -> pa.RecordBatchReader:
    """Convert a Polars LazyFrame into a RecordBatchReader.

    Returns
    -------
    pa.RecordBatchReader
        Reader over streamed record batches.
    """
    from codeintel.core.columnar.stream import LazyFrameStream

    stream = LazyFrameStream(
        frame,
        query_opt_flags=None,
        streaming=True,
        streaming_fallback=True,
        inspect=False,
    )
    return stream.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)


def arrow_reader_to_lazyframe(reader: pa.RecordBatchReader) -> pl.LazyFrame:
    """Convert an Arrow RecordBatchReader into a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame constructed from the Arrow reader.

    Raises
    ------
    ValueError
        If the Arrow reader fails to materialize and provides no schema.
    ArrowInvalid
        If the Arrow reader cannot be materialized and provides no schema.
    """
    try:
        dataset = pa_ds.dataset(reader)
    except (ValueError, pa.ArrowInvalid) as exc:
        schema = getattr(reader, "schema", None)
        if schema is None:
            raise
        empty = pa.Table.from_batches([], schema=schema)
        _ = exc
        return table_to_lazyframe(empty)
    return _coerce_goid_columns(pl.scan_pyarrow_dataset(dataset))


def table_to_lazyframe(table: pa.Table) -> pl.LazyFrame:
    """Convert an Arrow Table into a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame constructed from the Arrow table.
    """
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        return _coerce_goid_columns(frame.to_frame().lazy())
    return _coerce_goid_columns(frame.lazy())


def table_to_frame(table: pa.Table) -> pl.DataFrame:
    """Convert an Arrow Table into a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame constructed from the Arrow table.
    """
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        frame = frame.to_frame()
    return _coerce_goid_columns_frame(frame)


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

    Raises
    ------
    TypeError
        If the input type cannot be coerced into a DataFrame.
    """
    if isinstance(value, pl.DataFrame):
        result = _coerce_goid_columns_frame(value)
    elif isinstance(value, pl.LazyFrame):
        result = _coerce_goid_columns(value).collect()
    elif isinstance(value, pa.Table):
        result = table_to_frame(value)
    elif isinstance(value, pa.RecordBatchReader):
        result = arrow_reader_to_lazyframe(value).collect()
    elif isinstance(value, DuckDBRelation):
        result = arrow_reader_to_lazyframe(value.fetch_arrow_reader()).collect()
    elif isinstance(value, Iterable):
        reader = record_batch_reader_from_iterable(value, empty_policy="none")
        result = (
            pl.DataFrame()
            if reader is None
            else arrow_reader_to_lazyframe(reader).collect()
        )
    else:
        msg = f"Unsupported tabular input type: {type(value).__name__}"
        raise TypeError(msg)
    return result


def tabular_to_arrow_reader(value: InferableTabularInput) -> pa.RecordBatchReader:
    """Convert an inferable tabular input to a RecordBatchReader.

    Parameters
    ----------
    value
        Tabular input to convert.

    Returns
    -------
    pa.RecordBatchReader
        RecordBatchReader representation of the input.

    Raises
    ------
    TypeError
        If the input type cannot be coerced into a RecordBatchReader.

    Notes
    -----
    RecordBatchReader inputs are single-consume; materialize to a table or
    LazyFrame if reuse is required.
    """
    from codeintel.core.columnar.stream import coerce_arrow_reader

    reader: pa.RecordBatchReader | None = None
    if isinstance(value, pa.RecordBatchReader):
        reader = value
    elif isinstance(value, pa.Table):
        reader = table_to_reader(value)
    elif isinstance(value, DuckDBRelation):
        reader = value.fetch_arrow_reader()
    elif isinstance(value, pl.LazyFrame):
        reader = lazyframe_to_reader(value)
    elif isinstance(value, pl.DataFrame):
        reader = table_to_reader(value.to_arrow())
    else:
        reader = coerce_arrow_reader(value, batch_size=DEFAULT_ARROW_BATCH_SIZE)
        if reader is None and isinstance(value, Iterable):
            reader = _record_batch_reader_from_iterable(value)
            if reader is None:
                msg = "Unsupported tabular input type: empty iterable"
                raise TypeError(msg)
    if reader is not None:
        return reader
    msg = f"Unsupported tabular input type: {type(value).__name__}"
    raise TypeError(msg)


def tabular_to_arrow_table(value: InferableTabularInput) -> pa.Table:
    """Convert an inferable tabular input to an Arrow Table.

    Parameters
    ----------
    value
        Tabular input to convert.

    Returns
    -------
    pa.Table
        Arrow table representation of the input.

    Notes
    -----
    RecordBatchReader inputs are single-consume; avoid reusing them after
    calling this helper.
    """
    if isinstance(value, pa.Table):
        return value
    reader = tabular_to_arrow_reader(value)
    return reader_to_table(reader)


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

    return pa.RecordBatchReader.from_batches(first.schema, batch_iter())


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
        col
        for col in columns
        if isinstance(col, str) and _GOID_COLUMN_MARKER in col.lower()
    ]
    if not goid_columns:
        return frame
    return frame.with_columns(
        [pl.col(name).cast(_GOID_COLUMN_TYPE, strict=False) for name in goid_columns]
    )


def _coerce_goid_columns_frame(frame: pl.DataFrame) -> pl.DataFrame:
    goid_columns = [
        col
        for col in frame.columns
        if isinstance(col, str) and _GOID_COLUMN_MARKER in col.lower()
    ]
    if not goid_columns:
        return frame
    return frame.with_columns(
        [pl.col(name).cast(_GOID_COLUMN_TYPE, strict=False) for name in goid_columns]
    )


__all__ = [
    "arrow_reader_to_lazyframe",
    "lazyframe_to_reader",
    "reader_to_table",
    "record_batch_reader_from_iterable",
    "table_to_frame",
    "table_to_lazyframe",
    "table_to_reader",
    "tabular_to_arrow_reader",
    "tabular_to_arrow_table",
    "tabular_to_frame",
]
