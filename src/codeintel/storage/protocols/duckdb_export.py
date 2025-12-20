"""DuckDB adapters for export protocols."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa
from duckdb import DuckDBPyRelation

from codeintel.storage.protocols.export import ExportRelation, RecordBatch, RecordBatchReader

__all__ = [
    "ArrowRecordBatchReaderAdapter",
    "DuckDBRelationAdapter",
    "adapt_duckdb_relation",
]


@dataclass(frozen=True, slots=True)
class ArrowRecordBatchReaderAdapter:
    """Adapter to expose a PyArrow RecordBatchReader via the export protocol."""

    reader: pa.RecordBatchReader

    @property
    def schema(self) -> pa.Schema:
        """Return the PyArrow schema for this reader."""
        return self.reader.schema

    def __iter__(self) -> Iterator[RecordBatch]:
        """Iterate record batches."""
        return iter(self.reader)


@dataclass(frozen=True, slots=True)
class DuckDBRelationAdapter:
    """Adapter to expose a DuckDB relation via the export protocol."""

    relation: DuckDBPyRelation

    def fetch_record_batch(self, rows_per_batch: int) -> RecordBatchReader:
        """Return record batch reader for the relation."""
        reader = self.relation.fetch_record_batch(rows_per_batch)
        return ArrowRecordBatchReaderAdapter(reader)

    def aggregate(self, aggr_expr: str, group_expr: str = "") -> ExportRelation:
        """Return an aggregated relation for an expression."""
        aggregated = self.relation.aggregate(aggr_expr, group_expr)
        return DuckDBRelationAdapter(aggregated)

    def fetchone(self) -> tuple[object, ...] | None:
        """Return the next row, or None when no rows remain."""
        return self.relation.fetchone()

    def write_parquet(self, path: str) -> None:
        """Write the relation to a Parquet file."""
        self.relation.write_parquet(path)


def adapt_duckdb_relation(relation: DuckDBPyRelation) -> ExportRelation:
    """Wrap a DuckDB relation with the export protocol adapter."""
    return DuckDBRelationAdapter(relation)
