"""Helpers for building columnar ingestion frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import pyarrow as pa

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import lazyframe_from_rows, table_to_lazyframe
from codeintel.core.columnar.rows import columnar_row_count
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema


def empty_lazyframe_for_table(table_key: str) -> pl.LazyFrame:
    """Return an empty LazyFrame using the table schema.

    Returns
    -------
    pl.LazyFrame
        Empty LazyFrame with the table's schema applied.
    """
    schema = get_schema_service().require_table_schema(table_key)
    arrow_schema = arrow_schema_from_table_schema(table_schema=schema)
    table = pa.Table.from_batches([], schema=arrow_schema)
    return table_to_lazyframe(table)


def lazyframe_for_table(
    table_key: str,
    rows: Sequence[Sequence[object]],
) -> pl.LazyFrame:
    """Build a LazyFrame for table rows using the schema's column order.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with rows aligned to the schema's column order.
    """
    if not rows:
        return empty_lazyframe_for_table(table_key)
    schema = get_schema_service().require_table_schema(table_key)
    columns = tuple(schema.column_names())
    return lazyframe_from_rows(rows=rows, columns=columns)


def lazyframe_for_table_columns(
    table_key: str,
    columns: Mapping[str, Sequence[object]],
) -> pl.LazyFrame:
    """Build a LazyFrame for columnar data using the schema's column order.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with columns aligned to the schema order.

    Raises
    ------
    ValueError
        If input columns contain unexpected names.
    """
    if not columns:
        return empty_lazyframe_for_table(table_key)
    row_count = columnar_row_count(columns)
    if row_count == 0:
        return empty_lazyframe_for_table(table_key)
    schema = get_schema_service().require_table_schema(table_key)
    column_names = tuple(schema.column_names())
    extra = set(columns).difference(column_names)
    if extra:
        msg = f"Unexpected columns for {table_key}: {sorted(extra)}"
        raise ValueError(msg)
    ordered: dict[str, list[object]] = {}
    for name in column_names:
        values = columns.get(name)
        if values is None:
            ordered[name] = [None] * row_count
        else:
            ordered[name] = list(values)
    return pl.DataFrame(ordered).lazy()


def dedupe_frame_for_table(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    prefer_columns: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    """Deduplicate rows for a table based on its primary key.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with primary-key duplicates removed.
    """
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return frame
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [column for column in prefer_columns if column in set(schema.column_names())]
        if prefer:
            frame = frame.sort(by=prefer, descending=[True] * len(prefer), nulls_last=True)
    return frame.unique(subset=key_columns, keep="first")


__all__ = [
    "dedupe_frame_for_table",
    "empty_lazyframe_for_table",
    "lazyframe_for_table",
    "lazyframe_for_table_columns",
]
