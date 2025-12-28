"""Helpers for building columnar ingestion frames."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pyarrow as pa

from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.conversion import lazyframe_from_rows, table_to_lazyframe
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema


def empty_lazyframe_for_table(table_key: str) -> pl.LazyFrame:
    """Return an empty LazyFrame using the table schema."""
    schema = get_schema_service().require_table_schema(table_key)
    arrow_schema = arrow_schema_from_table_schema(table_schema=schema)
    table = pa.Table.from_batches([], schema=arrow_schema)
    return table_to_lazyframe(table)


def lazyframe_for_table(
    table_key: str,
    rows: Sequence[Sequence[object]],
) -> pl.LazyFrame:
    """Build a LazyFrame for table rows using the schema's column order."""
    if not rows:
        return empty_lazyframe_for_table(table_key)
    schema = get_schema_service().require_table_schema(table_key)
    columns = tuple(schema.column_names())
    return lazyframe_from_rows(rows=rows, columns=columns)


def dedupe_frame_for_table(
    frame: pl.LazyFrame,
    *,
    table_key: str,
    prefer_columns: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    """Deduplicate rows for a table based on its primary key."""
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return frame
    key_columns = list(schema.primary_key)
    if prefer_columns:
        prefer = [column for column in prefer_columns if column in set(schema.column_names())]
        if prefer:
            frame = frame.sort(by=prefer, descending=[True] * len(prefer), nulls_last=True)
    return frame.unique(subset=key_columns, keep="first")


__all__ = ["dedupe_frame_for_table", "empty_lazyframe_for_table", "lazyframe_for_table"]
