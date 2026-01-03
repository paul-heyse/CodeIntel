"""Deprecated wrapper for ingestion frame helpers.

Use codeintel.build.tabular.frames instead.
"""

from __future__ import annotations

import polars as pl

from codeintel.build.tabular.frames import (
    ColumnsSpec,
    dedupe_frame_for_table,
    empty_frame_for_table,
    lazyframe_for_ingest_columns,
    lazyframe_for_table_columns,
)


def empty_lazyframe_for_table(table_key: str) -> pl.LazyFrame:
    """Return an empty LazyFrame using the table schema.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame aligned to the table schema.
    """
    return empty_frame_for_table(table_key)


__all__ = [
    "ColumnsSpec",
    "dedupe_frame_for_table",
    "empty_lazyframe_for_table",
    "lazyframe_for_ingest_columns",
    "lazyframe_for_table_columns",
]
