"""Columnar conversion helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds

from codeintel.build.tabular.types import InferableTabularInput, TabularRelation

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
        batches = list(value)
        if not batches:
            return pl.DataFrame().lazy()
        reader = pa.RecordBatchReader.from_batches(batches[0].schema, batches)
        return arrow_reader_to_lazyframe(reader)
    msg = f"Unsupported tabular input type: {type(value).__name__}"
    raise TypeError(msg)


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


__all__ = [
    "arrow_reader_to_lazyframe",
    "relation_to_arrow_reader",
    "relation_to_polars_lazy",
    "table_to_lazyframe",
    "tabular_to_lazyframe",
]
