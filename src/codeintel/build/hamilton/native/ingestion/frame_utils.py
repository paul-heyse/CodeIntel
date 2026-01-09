"""Deprecated wrapper for ingestion frame helpers.

These helpers now return Arrow tables to avoid streaming outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from codeintel.core.columnar.conversion import empty_table_from_schema
from codeintel.core.columnar.rows import (
    columnar_row_count,
    empty_table_for_table,
    table_for_columnar_rows,
)

ColumnsSpec = Mapping[str, Sequence[object]] | Sequence[str] | None


def _normalize_columns(columns: ColumnsSpec) -> Mapping[str, Sequence[object]]:
    if columns is None:
        return {}
    if isinstance(columns, Mapping):
        return columns
    return {str(name): [] for name in columns}


def dedupe_frame_for_table(_table_key: str, frame: pa.Table) -> pa.Table:
    """Return the input frame unchanged for Arrow-first ingestion.

    Returns
    -------
    pyarrow.Table
        Dedupe is handled upstream; this is a passthrough for compatibility.
    """
    return frame


def lazyframe_for_ingest_columns(table_key: str, rows: Mapping[str, Sequence[object]]) -> pa.Table:
    """Return an Arrow table for ingest columnar rows.

    Returns
    -------
    pyarrow.Table
        Table aligned to the provided columnar rows.
    """
    table, _ = table_for_columnar_rows(table_key, rows)
    return table


def lazyframe_for_table_columns(
    table_key: str,
    columns: ColumnsSpec,
) -> pa.Table:
    """Return an Arrow table aligned to the table schema.

    Returns
    -------
    pyarrow.Table
        Table aligned to the table schema for the requested columns.
    """
    normalized = _normalize_columns(columns)
    if not normalized or columnar_row_count(normalized) == 0:
        try:
            return empty_table_for_table(table_key)
        except (KeyError, RuntimeError):
            return empty_table_from_schema(pa.schema([]))
    table, _ = table_for_columnar_rows(table_key, normalized)
    return table


def empty_lazyframe_for_table(table_key: str) -> pa.Table:
    """Return an empty Arrow table using the table schema.

    Returns
    -------
    pyarrow.Table
        Empty table aligned to the table schema.
    """
    return empty_table_for_table(table_key)


__all__ = [
    "ColumnsSpec",
    "dedupe_frame_for_table",
    "empty_lazyframe_for_table",
    "lazyframe_for_ingest_columns",
    "lazyframe_for_table_columns",
]
