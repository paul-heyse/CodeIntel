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
    "ResultStream",
]


class RecordBatch(Protocol):
    """Protocol for record batches emitted by export relations."""

    num_rows: int
    num_columns: int

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the batch.

        Returns
        -------
        pyarrow.Schema
            Schema describing the batch columns.
        """
        ...

    def column(self, i: int) -> pa.Array:
        """Return the i-th column array.

        Parameters
        ----------
        i
            Column index.

        Returns
        -------
        pyarrow.Array
            Column array at the requested index.
        """
        ...


class RecordBatchReader(Protocol):
    """Protocol describing a record batch reader with schema metadata."""

    @property
    def schema(self) -> pa.Schema:
        """Return the schema for the record batch stream."""
        ...

    def __iter__(self) -> Iterator[RecordBatch]:
        """Iterate record batches."""
        ...


class ResultStream(Protocol):
    """Protocol for streaming result readers without eager materialization."""

    def to_reader(self, *, batch_size: int = DEFAULT_ARROW_BATCH_SIZE) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for streaming results."""
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
