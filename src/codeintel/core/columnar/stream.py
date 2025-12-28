"""Columnar stream protocol and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

if TYPE_CHECKING:
    import polars as pl

    type PolarsLazyFrame = pl.LazyFrame
else:
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


@runtime_checkable
class ColumnarStream(Protocol):
    """Protocol for columnar streaming sources."""

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the stream."""
        ...

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for the stream."""
        ...

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Return a Polars LazyFrame for the stream."""
        ...

    def to_table(self) -> pa.Table:
        """Return a fully materialized Arrow table (last resort)."""
        ...


@dataclass(frozen=True, slots=True)
class RecordBatchReaderStream:
    """ColumnarStream adapter for Arrow RecordBatchReader."""

    reader: pa.RecordBatchReader

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the stream."""
        return self.reader.schema

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return the underlying reader (batch_size is advisory only)."""
        _ = batch_size
        return self.reader

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Convert the stream into a Polars LazyFrame."""
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame conversion"
            raise RuntimeError(msg)
        frame = pl.from_arrow(self.reader)
        if isinstance(frame, pl.Series):
            return frame.to_frame().lazy()
        return frame.lazy()

    def to_table(self) -> pa.Table:
        """Materialize the stream into a table."""
        return pa.Table.from_batches(list(self.reader), schema=self.reader.schema)


@dataclass(frozen=True, slots=True)
class LazyFrameStream:
    """ColumnarStream adapter for Polars LazyFrame."""

    lazyframe: PolarsLazyFrame

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the stream."""
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame schema"
            raise RuntimeError(msg)
        if isinstance(self.lazyframe, pl.LazyFrame):
            return self.lazyframe.collect_schema().to_arrow()
        msg = "LazyFrameStream expects a polars.LazyFrame"
        raise TypeError(msg)

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Stream the LazyFrame as record batches."""
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame streaming"
            raise RuntimeError(msg)
        if not isinstance(self.lazyframe, pl.LazyFrame):
            msg = "LazyFrameStream expects a polars.LazyFrame"
            raise TypeError(msg)
        batches = self.lazyframe.collect_batches(batch_size=batch_size, engine="streaming")
        return pa.RecordBatchReader.from_batches(self.schema, batches)

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Return the underlying LazyFrame."""
        return self.lazyframe

    def to_table(self) -> pa.Table:
        """Materialize the LazyFrame into a table (last resort)."""
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame materialization"
            raise RuntimeError(msg)
        if not isinstance(self.lazyframe, pl.LazyFrame):
            msg = "LazyFrameStream expects a polars.LazyFrame"
            raise TypeError(msg)
        return self.lazyframe.collect(engine="streaming").to_arrow()


ColumnarStreamAdapter = RecordBatchReaderStream | LazyFrameStream

__all__ = [
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "RecordBatchReaderStream",
]
