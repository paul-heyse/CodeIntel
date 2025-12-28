"""Canonical tabular type aliases for relation-first compute."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from codeintel.storage.duckdb_types import DuckDBRelation

type TabularRelation = DuckDBRelation
type TabularFrame = pl.LazyFrame
type TabularInput = DuckDBRelation | pa.RecordBatchReader | pa.Table | TabularFrame
type InferableTabularInput = pa.RecordBatchReader | pa.Table | TabularFrame

__all__ = ["InferableTabularInput", "TabularFrame", "TabularInput", "TabularRelation"]
