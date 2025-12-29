"""Canonical tabular type aliases for relation-first compute."""

from __future__ import annotations

import pyarrow as pa

from codeintel.core.columnar.tabular_adapter import TabularFrame, TabularInput, TabularRelation

type InferableTabularInput = pa.RecordBatchReader | pa.Table | TabularFrame

__all__ = ["InferableTabularInput", "TabularFrame", "TabularInput", "TabularRelation"]
