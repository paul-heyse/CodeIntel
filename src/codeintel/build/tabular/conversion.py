"""Columnar conversion helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.types import InferableTabularInput, TabularRelation


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
    """
    try:
        table = reader.read_all()
    except (ValueError, pa.ArrowInvalid) as exc:
        schema = getattr(reader, "schema", None)
        if schema is None:
            raise
        table = pa.Table.from_batches([], schema=schema)
        _ = exc
    return table_to_lazyframe(table)


def table_to_lazyframe(table: pa.Table) -> pl.LazyFrame:
    """Convert an Arrow Table into a Polars LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame constructed from the Arrow table.
    """
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


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
        return value
    if isinstance(value, pl.DataFrame):
        return value.lazy()
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


__all__ = [
    "arrow_reader_to_lazyframe",
    "relation_to_arrow_reader",
    "relation_to_polars_lazy",
    "table_to_lazyframe",
    "tabular_to_lazyframe",
]
