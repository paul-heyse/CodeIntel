"""Columnar conversion helpers for build pipelines."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.types import TabularRelation


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
    frame = pl.from_arrow(reader)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


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


__all__ = [
    "arrow_reader_to_lazyframe",
    "relation_to_arrow_reader",
    "relation_to_polars_lazy",
    "table_to_lazyframe",
]
