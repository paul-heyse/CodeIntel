"""Canonical tabular type aliases for relation-first compute."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
import pyarrow as pa

from codeintel.storage.duckdb_types import DuckDBRelation

type RecordBatchIterable = Iterable[pa.RecordBatch]
type TabularRelation = DuckDBRelation
type TabularFrame = pl.LazyFrame
type TabularInput = DuckDBRelation | pa.RecordBatchReader | pa.Table | TabularFrame
type InferableTabularInput = pa.RecordBatchReader | pa.Table | TabularFrame | RecordBatchIterable

__all__ = [
    "InferableTabularInput",
    "RecordBatchIterable",
    "TabularFrame",
    "TabularInput",
    "TabularRelation",
]
