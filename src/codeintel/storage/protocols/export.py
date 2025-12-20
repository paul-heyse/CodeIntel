"""DuckDB-agnostic protocols for export relations and record batches."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import pyarrow as pa

from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE

__all__ = [
    "ExportRelation",
    "RecordBatch",
    "RecordBatchReader",
]


class RecordBatch(Protocol):
    """Protocol for record batches emitted by export relations."""

    num_rows: int

    def to_pydict(self) -> dict[str, list[object]]:
        """Return a columnar mapping for the batch."""
        ...


class RecordBatchReader(Protocol):
    """Protocol describing a record batch reader with schema metadata."""

    schema: pa.Schema

    def __iter__(self) -> Iterator[RecordBatch]:
        """Iterate record batches."""
        ...


class ExportRelation(Protocol):
    """Protocol surface for export-ready relations."""

    def fetch_record_batch(
        self, rows_per_batch: int = DEFAULT_ARROW_BATCH_SIZE
    ) -> RecordBatchReader:
        """Return an iterator of record batches."""
        ...

    def aggregate(self, aggr_expr: str, group_expr: str = "") -> ExportRelation:
        """Return an aggregated relation for an expression."""
        ...

    def fetchone(self) -> tuple[object, ...] | None:
        """Return the next row, or None when no rows remain."""
        ...
