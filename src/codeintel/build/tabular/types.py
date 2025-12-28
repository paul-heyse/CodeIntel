"""Canonical tabular type aliases for relation-first compute."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from codeintel.storage.duckdb_types import DuckDBRelation

type TabularRelation = DuckDBRelation
type TabularInput = DuckDBRelation | pa.Table | pa.RecordBatchReader | pl.DataFrame | pl.LazyFrame

__all__ = ["TabularInput", "TabularRelation"]
