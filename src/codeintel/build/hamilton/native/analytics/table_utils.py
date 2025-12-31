"""Shared helpers for analytics table construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import pyarrow as pa

from codeintel.build.schemas import get_schema_provider
from codeintel.build.tabular.conversion import table_to_lazyframe
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema


def empty_frame_for_table(table_key: str) -> pl.LazyFrame:
    """Return an empty LazyFrame matching the table schema.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame with the table's column names and types.
    """
    schema = get_schema_provider().require_table_schema(table_key)
    arrow_schema = arrow_schema_from_table_schema(table_schema=schema)
    table = pa.Table.from_batches([], schema=arrow_schema)
    return table_to_lazyframe(table)


def _columns_for_table(table_key: str) -> list[str]:
    schema = get_schema_provider().require_table_schema(table_key)
    return [column.name for column in schema.columns]


def rows_to_frame(
    table_key: str,
    rows: Sequence[Mapping[str, object]] | Sequence[Sequence[object]],
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Convert row sequences into a LazyFrame with schema-ordered columns.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).
    rows
        Row mappings or row tuples in the expected column order.
    columns
        Optional explicit column order for tuple rows.

    Returns
    -------
    polars.LazyFrame
        LazyFrame with schema-ordered columns.
    """
    if not rows:
        return empty_frame_for_table(table_key)
    ordered_columns = list(columns or _columns_for_table(table_key))
    first = rows[0]
    if isinstance(first, Mapping):
        frame = pl.DataFrame(rows)
        missing = [col for col in ordered_columns if col not in frame.columns]
        if missing:
            frame = frame.with_columns([pl.lit(None).alias(col) for col in missing])
        return frame.lazy().select(ordered_columns)
    frame = pl.DataFrame(rows, schema=ordered_columns, orient="row")
    return frame.lazy().select(ordered_columns)


__all__ = ["empty_frame_for_table", "rows_to_frame"]
