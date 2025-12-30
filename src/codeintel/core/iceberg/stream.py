"""Iceberg-backed ColumnarStream implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow as pa

from codeintel.core.columnar.tabular_adapter import ColumnarStream, PolarsLazyFrame

if TYPE_CHECKING:
    from pyiceberg.table import DataScan

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


@dataclass(frozen=True, slots=True)
class IcebergColumnarStream(ColumnarStream):
    """ColumnarStream adapter for Iceberg DataScan objects."""

    scan: DataScan

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the scan.

        Returns
        -------
        pyarrow.Schema
            Arrow schema for the scan output.
        """
        return self.scan.to_arrow_batch_reader().schema

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for the scan.

        Returns
        -------
        pyarrow.RecordBatchReader
            RecordBatchReader for the scan output.
        """
        _ = batch_size
        return self.scan.to_arrow_batch_reader()

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Return a Polars LazyFrame for the scan.

        Returns
        -------
        polars.LazyFrame
            LazyFrame representation of the scan output.
        """
        frame = self.scan.to_polars()
        if pl is not None and isinstance(frame, pl.DataFrame):
            return frame.lazy()
        return cast("PolarsLazyFrame", frame)

    def to_table(self) -> pa.Table:
        """Materialize the scan into an Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized Arrow table for the scan output.
        """
        return self.scan.to_arrow()


__all__ = ["IcebergColumnarStream"]
