"""Columnar conversion helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds

from codeintel.build.tabular.types import InferableTabularInput, TabularRelation
from codeintel.core.columnar import LazyFrameStream
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.duckdb_types import DuckDBRelation

_GOID_COLUMN_MARKER = "goid_h128"
_GOID_COLUMN_TYPE = pl.Decimal(38, 0)


def relation_to_arrow_reader(relation: TabularRelation) -> pa.RecordBatchReader:
    """Return a streaming Arrow reader for a DuckDB relation.

    Returns
    -------
    pa.RecordBatchReader
        Arrow record batch reader for the relation.
    """
    return relation.fetch_arrow_reader()


def relation_to_polars_lazy(relation: TabularRelation) -> pl.LazyFrame:
    """Return a Polars LazyFrame derived from a DuckDB relation.

    Returns
    -------
    pl.LazyFrame
        LazyFrame backed by the relation's Arrow stream.
    """
    reader = relation_to_arrow_reader(relation)
    return arrow_reader_to_lazyframe(reader)


def table_to_reader(table: pa.Table) -> pa.RecordBatchReader:
    """Convert an Arrow Table into a RecordBatchReader.

    Returns
    -------
    pa.RecordBatchReader
        Reader over record batches from the table.
    """
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def lazyframe_to_reader(frame: pl.LazyFrame) -> pa.RecordBatchReader:
    """Convert a Polars LazyFrame into a RecordBatchReader.

    Returns
    -------
    pa.RecordBatchReader
        Reader over streamed record batches.
    """
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
    """
    if isinstance(value, pa.RecordBatchReader):
        return value
    if isinstance(value, pa.Table):
        return table_to_reader(value)
    if isinstance(value, DuckDBRelation):
        return relation_to_arrow_reader(value)
    if isinstance(value, pl.LazyFrame):
        return lazyframe_to_reader(value)
    if isinstance(value, pl.DataFrame):
        table = value.to_arrow()
        return table_to_reader(table)
    if isinstance(value, Iterable):
        reader = _record_batch_reader_from_iterable(value)
        if reader is None:
            msg = "Unsupported tabular input type: empty iterable"
            raise TypeError(msg)
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
    """
    if isinstance(value, pa.Table):
        return value
    reader = tabular_to_arrow_reader(value)
    return pa.Table.from_batches(list(reader), schema=reader.schema)


def tabular_to_lazyframe(value: InferableTabularInput) -> pl.LazyFrame:
    """Convert an inferable tabular input to a Polars LazyFrame.

    Parameters
    ----------
    value
        Tabular input to convert.

    Returns
    -------
    pl.LazyFrame
        LazyFrame representation of the input.

    Raises
    ------
    TypeError
        If the input type cannot be coerced into a LazyFrame.
    """
    if isinstance(value, pl.LazyFrame):
        return _coerce_goid_columns(value)
    if isinstance(value, pl.DataFrame):
        return _coerce_goid_columns(value.lazy())
    if isinstance(value, pa.Table):
        return table_to_lazyframe(value)
    if isinstance(value, pa.RecordBatchReader):
        return arrow_reader_to_lazyframe(value)
    if isinstance(value, Iterable):
        reader = _record_batch_reader_from_iterable(value)
        if reader is None:
            return pl.DataFrame().lazy()
        return arrow_reader_to_lazyframe(reader)
    msg = f"Unsupported tabular input type: {type(value).__name__}"
    raise TypeError(msg)


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
        return _coerce_goid_columns_frame(value)
    if isinstance(value, pl.LazyFrame):
        return _coerce_goid_columns(value).collect()
    if isinstance(value, pa.Table):
        return table_to_frame(value)
    if isinstance(value, pa.RecordBatchReader):
        return arrow_reader_to_lazyframe(value).collect()
    if isinstance(value, Iterable):
        reader = _record_batch_reader_from_iterable(value)
        if reader is None:
            return pl.DataFrame()
        return arrow_reader_to_lazyframe(reader).collect()
    msg = f"Unsupported tabular input type: {type(value).__name__}"
    raise TypeError(msg)


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
    "lazyframe_to_reader",
    "relation_to_arrow_reader",
    "relation_to_polars_lazy",
    "table_to_lazyframe",
    "table_to_reader",
    "tabular_to_arrow_reader",
    "tabular_to_arrow_table",
    "tabular_to_frame",
    "tabular_to_lazyframe",
]
