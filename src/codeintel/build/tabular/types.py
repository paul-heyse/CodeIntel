"""Canonical tabular type aliases for relation-first compute."""

from __future__ import annotations

import duckdb
import polars as pl
import pyarrow as pa

type TabularRelation = duckdb.DuckDBPyRelation
type TabularInput = (
    duckdb.DuckDBPyRelation | pa.Table | pa.RecordBatchReader | pl.DataFrame | pl.LazyFrame
)

__all__ = ["TabularInput", "TabularRelation"]
