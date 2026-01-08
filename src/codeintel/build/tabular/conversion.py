"""Columnar conversion helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl
import pyarrow as pa

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.types import InferableTabularInput, TabularRelation
from codeintel.core.columnar.conversion import (
    arrow_reader_to_lazyframe,
    lazyframe_to_reader,
    reader_to_table,
    record_batch_reader_from_iterable,
    table_to_lazyframe,
    table_to_reader,
)
from codeintel.core.columnar.conversion import (
    table_to_frame as _core_table_to_frame,
)
from codeintel.core.columnar.conversion import (
    tabular_to_arrow_reader as _tabular_to_arrow_reader,
)
from codeintel.core.columnar.conversion import (
    tabular_to_arrow_table as _tabular_to_arrow_table,
)
from codeintel.core.columnar.conversion import (
    tabular_to_frame as _core_tabular_to_frame,
)
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

    Notes
    -----
    RecordBatchReader inputs are single-consume; materialize to a table or
    LazyFrame if reuse is required.
    """
    return _tabular_to_arrow_reader(value)


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
    return _tabular_to_arrow_table(value)


def tabular_to_scoped_table(
    value: InferableTabularInput,
    *,
    columns: Sequence[str] | None,
    scope: SnapshotScope | None,
    require_scope_columns: bool,
) -> pa.Table:
    """Convert an inferable tabular input into a scoped Arrow table.

    Returns
    -------
    pa.Table
        Arrow table projected to columns and filtered by snapshot scope.
    """
    table = tabular_to_arrow_table(value)
    if scope is not None:
        table = scope.filter_arrow_table(table, require_columns=require_scope_columns)
    if columns is not None:
        table = table.select(list(columns))
    return table


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
        result = _coerce_goid_columns(value)
    elif isinstance(value, pl.DataFrame):
        result = _coerce_goid_columns(value.lazy())
    elif isinstance(value, pa.Table):
        result = table_to_lazyframe(value)
    elif isinstance(value, pa.RecordBatchReader):
        result = arrow_reader_to_lazyframe(value)
    elif isinstance(value, DuckDBRelation):
        result = arrow_reader_to_lazyframe(value.fetch_arrow_reader())
    elif isinstance(value, Iterable):
        reader = record_batch_reader_from_iterable(value, empty_policy="none")
        result = pl.DataFrame().lazy() if reader is None else arrow_reader_to_lazyframe(reader)
    else:
        msg = f"Unsupported tabular input type: {type(value).__name__}"
        raise TypeError(msg)
    return result


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
    return _core_tabular_to_frame(value)


def table_to_frame(table: pa.Table) -> pl.DataFrame:
    """Convert an Arrow Table into a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        DataFrame constructed from the Arrow table.
    """
    return _core_table_to_frame(table)


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


__all__ = [
    "arrow_reader_to_lazyframe",
    "lazyframe_to_reader",
    "reader_to_table",
    "record_batch_reader_from_iterable",
    "relation_to_arrow_reader",
    "relation_to_polars_lazy",
    "table_to_frame",
    "table_to_lazyframe",
    "table_to_reader",
    "tabular_to_arrow_reader",
    "tabular_to_arrow_table",
    "tabular_to_frame",
    "tabular_to_lazyframe",
    "tabular_to_scoped_table",
]
