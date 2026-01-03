"""DuckDB-agnostic protocols for export relations and record batches (core re-export)."""

from __future__ import annotations

from codeintel.core.ports.export import (
    ExportRelation,
    RecordBatch,
    RecordBatchReader,
    ResultStream,
)

__all__ = [
    "ExportRelation",
    "RecordBatch",
    "RecordBatchReader",
    "ResultStream",
]
