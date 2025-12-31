"""Shared helpers for analytics table construction."""

from __future__ import annotations

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


__all__ = ["empty_frame_for_table"]
