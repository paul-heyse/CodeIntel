"""Canonical tabular type aliases for inferable compute nodes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from codeintel.core.duckdb_types import DuckDBRelation

if TYPE_CHECKING:
    type RecordBatchIterable = Iterable[pa.RecordBatch]
    type TabularRelation = DuckDBRelation
    type TabularFrame = pl.LazyFrame
    # RecordBatchReader inputs are single-consume; materialize if reuse is required.
    type InferableTabularInput = (
        pa.RecordBatchReader
        | pa.Table
        | pl.DataFrame
        | TabularFrame
        | RecordBatchIterable
        | DuckDBRelation
    )
    type TabularInput = InferableTabularInput
    type TabularInputWithRelation = TabularInput | TabularRelation
else:
    RecordBatchIterable = Iterable[pa.RecordBatch]
    TabularRelation = DuckDBRelation
    TabularFrame = pl.LazyFrame
    # RecordBatchReader inputs are single-consume; materialize if reuse is required.
    InferableTabularInput = (
        pa.RecordBatchReader
        | pa.Table
        | pl.DataFrame
        | TabularFrame
        | RecordBatchIterable
        | DuckDBRelation
    )
    TabularInput = InferableTabularInput
    TabularInputWithRelation = TabularInput | TabularRelation

__all__ = [
    "InferableTabularInput",
    "RecordBatchIterable",
    "TabularFrame",
    "TabularInput",
    "TabularInputWithRelation",
    "TabularRelation",
]
