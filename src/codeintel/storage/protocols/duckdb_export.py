"""DuckDB adapters for export protocols."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa
from duckdb import DuckDBPyRelation

from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
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
        """Return the PyArrow schema for this reader.

        Returns
        -------
        pa.Schema
            Schema describing the record batch stream.
        """
        return self.reader.schema

    def __iter__(self) -> Iterator[RecordBatch]:
        """Iterate record batches.

        Returns
        -------
        Iterator[RecordBatch]
            Iterator over record batches.
        """
        return iter(self.reader)


@dataclass(frozen=True, slots=True)
class DuckDBRelationAdapter:
    """Adapter to expose a DuckDB relation via the export protocol."""

    relation: DuckDBPyRelation

    def fetch_record_batch(
        self, rows_per_batch: int = DEFAULT_ARROW_BATCH_SIZE
    ) -> RecordBatchReader:
        """Return record batch reader for the relation.

        Parameters
        ----------
        rows_per_batch
            Maximum rows per batch.

        Returns
        -------
        RecordBatchReader
            Reader yielding record batches.
        """
        reader = self.relation.fetch_record_batch(rows_per_batch)
        return ArrowRecordBatchReaderAdapter(reader)

    def aggregate(self, aggr_expr: str, group_expr: str = "") -> ExportRelation:
        """Return an aggregated relation for an expression.

        Parameters
        ----------
        aggr_expr
            Aggregation expression.
        group_expr
            Optional grouping expression.

        Returns
        -------
        ExportRelation
            Aggregated relation adapter.
        """
        aggregated = self.relation.aggregate(aggr_expr, group_expr)
        return DuckDBRelationAdapter(aggregated)

    def fetchone(self) -> tuple[object, ...] | None:
        """Return the next row, or None when no rows remain.

        Returns
        -------
        tuple[object, ...] | None
            Next row, or None when no rows remain.
        """
        return self.relation.fetchone()

    def write_parquet(self, path: str) -> None:
        """Write the relation to a Parquet file.

        Parameters
        ----------
        path
            Destination path for the parquet file.
        """
        self.relation.write_parquet(path)


def adapt_duckdb_relation(relation: DuckDBPyRelation) -> ExportRelation:
    """Wrap a DuckDB relation with the export protocol adapter.

    Parameters
    ----------
    relation
        DuckDB relation to adapt.

    Returns
    -------
    ExportRelation
        Adapted relation exposing the export protocol.
    """
    return DuckDBRelationAdapter(relation)
