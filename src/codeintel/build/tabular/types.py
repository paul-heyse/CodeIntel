"""Canonical tabular type aliases for relation-first compute."""

from __future__ import annotations

from typing import TypeAlias

import duckdb
import polars as pl
import pyarrow as pa

TabularRelation: TypeAlias = duckdb.DuckDBPyRelation
TabularInput: TypeAlias = (
    duckdb.DuckDBPyRelation | pa.Table | pa.RecordBatchReader | pl.DataFrame | pl.LazyFrame
)

__all__ = ["TabularInput", "TabularRelation"]
