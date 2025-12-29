"""Compatibility wrapper for Arrow/Polars schema conversion helpers."""

from __future__ import annotations

from codeintel.core.schemas.contracts import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_dataframe,
    table_schema_from_polars_lazyframe,
    table_schema_from_polars_schema,
)

__all__ = [
    "table_schema_from_arrow_schema",
    "table_schema_from_polars_dataframe",
    "table_schema_from_polars_lazyframe",
    "table_schema_from_polars_schema",
]
