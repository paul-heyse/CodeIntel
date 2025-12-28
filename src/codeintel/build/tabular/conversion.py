"""Columnar conversion helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.types import TabularRelation


def relation_to_arrow_reader(relation: TabularRelation) -> pa.RecordBatchReader:
    """Return a streaming Arrow reader for a DuckDB relation."""
    return relation.fetch_arrow_reader()


def relation_to_polars_lazy(relation: TabularRelation) -> pl.LazyFrame:
    """Return a Polars LazyFrame derived from a DuckDB relation."""
    reader = relation_to_arrow_reader(relation)
    return arrow_reader_to_lazyframe(reader)


def arrow_reader_to_lazyframe(reader: pa.RecordBatchReader) -> pl.LazyFrame:
    """Convert an Arrow RecordBatchReader into a Polars LazyFrame."""
    frame = pl.from_arrow(reader)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


def table_to_lazyframe(table: pa.Table) -> pl.LazyFrame:
    """Convert an Arrow Table into a Polars LazyFrame."""
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


def lazyframe_from_rows(
    *,
    rows: Sequence[Sequence[object]],
    columns: Sequence[str],
) -> pl.LazyFrame:
    """Build a Polars LazyFrame from row tuples and column names."""
    frame = pl.DataFrame(rows, schema=list(columns))
    return frame.lazy()


__all__ = [
    "arrow_reader_to_lazyframe",
    "lazyframe_from_rows",
    "relation_to_arrow_reader",
    "relation_to_polars_lazy",
    "table_to_lazyframe",
]
