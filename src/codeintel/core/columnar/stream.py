"""Columnar stream protocol and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.schema import unify_schema_for_batches
from codeintel.core.columnar.tabular_adapter import (
    ColumnarStream,
    PolarsExecutionOptions,
    collect_batches,
    collect_lazyframe,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars import LazyFrame

    type PolarsLazyFrame = LazyFrame
else:
    type PolarsLazyFrame = object

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None
try:
    from polars.exceptions import PolarsError
except ImportError:  # pragma: no cover
    PolarsError = Exception


@dataclass(frozen=True, slots=True)
class RecordBatchReaderStream:
    """ColumnarStream adapter for Arrow RecordBatchReader."""

    reader: pa.RecordBatchReader

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the stream.

        Returns
        -------
        pyarrow.Schema
            Schema describing the stream output.
        """
        return self.reader.schema

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return the underlying reader (batch_size is advisory only).

        Parameters
        ----------
        batch_size
            Target batch size (not enforced for Arrow readers).

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the stream batches.
        """
        _ = batch_size
        return self.reader

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Convert the stream into a Polars LazyFrame.

        Returns
        -------
        polars.LazyFrame
            LazyFrame view over the stream data.

        Raises
        ------
        RuntimeError
            If Polars is unavailable.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame conversion"
            raise RuntimeError(msg)
        frame = pl.from_arrow(self.reader)
        if isinstance(frame, pl.Series):
            return frame.to_frame().lazy()
        return frame.lazy()

    def to_table(self) -> pa.Table:
        """Materialize the stream into a table.

        Returns
        -------
        pyarrow.Table
            Materialized table containing the stream data.
        """
        batches = list(self.reader)
        schema = unify_schema_for_batches(batches, base_schema=self.reader.schema)
        return pa.Table.from_batches(batches, schema=schema)


@dataclass(frozen=True, slots=True)
class LazyFrameStream:
    """ColumnarStream adapter for Polars LazyFrame."""

    lazyframe: PolarsLazyFrame
    query_opt_flags: object | None = None
    streaming: bool = True
    streaming_fallback: bool = True
    inspect: bool = False

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the stream.

        Returns
        -------
        pyarrow.Schema
            Schema describing the stream output.

        Raises
        ------
        RuntimeError
            If Polars is unavailable.
        TypeError
            If the wrapped object is not a LazyFrame.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame schema"
            raise RuntimeError(msg)
        if isinstance(self.lazyframe, pl.LazyFrame):
            return self.lazyframe.collect_schema().to_arrow()
        msg = "LazyFrameStream expects a polars.LazyFrame"
        raise TypeError(msg)

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Stream the LazyFrame as record batches.

        Parameters
        ----------
        batch_size
            Target row chunk size for streaming batches.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the streamed record batches.

        Raises
        ------
        ValueError
            If batch_size is not positive.
        RuntimeError
            If Polars is unavailable.
        TypeError
            If the wrapped object is not a LazyFrame.
        """
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame streaming"
            raise RuntimeError(msg)
        if not isinstance(self.lazyframe, pl.LazyFrame):
            msg = "LazyFrameStream expects a polars.LazyFrame"
            raise TypeError(msg)

        def _iter_batches() -> Iterator[pa.RecordBatch]:
            if self.inspect:
                _maybe_inspect(self.lazyframe)
            streaming = self.streaming
            try:
                yield from _collect_batches(
                    self.lazyframe,
                    batch_size=batch_size,
                    streaming=streaming,
                    query_opt_flags=self.query_opt_flags,
                )
            except PolarsError:
                if streaming and self.streaming_fallback:
                    yield from _collect_batches(
                        self.lazyframe,
                        batch_size=batch_size,
                        streaming=False,
                        query_opt_flags=self.query_opt_flags,
                    )
                else:
                    raise

        return pa.RecordBatchReader.from_batches(self.schema, _iter_batches())

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Return the underlying LazyFrame.

        Returns
        -------
        polars.LazyFrame
            LazyFrame backing this stream adapter.
        """
        return self.lazyframe

    def to_table(self) -> pa.Table:
        """Materialize the LazyFrame into a table (last resort).

        Returns
        -------
        pyarrow.Table
            Materialized table containing the LazyFrame results.

        Raises
        ------
        RuntimeError
            If Polars is unavailable.
        TypeError
            If the wrapped object is not a LazyFrame.
        """
        if pl is None:  # pragma: no cover
            msg = "polars is required for LazyFrame materialization"
            raise RuntimeError(msg)
        if not isinstance(self.lazyframe, pl.LazyFrame):
            msg = "LazyFrameStream expects a polars.LazyFrame"
            raise TypeError(msg)
        if self.inspect:
            _maybe_inspect(self.lazyframe)
        options = PolarsExecutionOptions(
            streaming=self.streaming,
            query_opt_flags=self.query_opt_flags,
            inspect=self.inspect,
            streaming_fallback=self.streaming_fallback,
        )
        return collect_lazyframe(self.lazyframe, options=options).to_arrow()


def _collect_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    streaming: bool,
    query_opt_flags: object | None,
) -> Iterator[pa.RecordBatch]:
    options = PolarsExecutionOptions(
        streaming=streaming,
        query_opt_flags=query_opt_flags,
    )
    result = collect_batches(
        lazyframe,
        batch_size=batch_size,
        options=options,
    )
    for frame in result:
        table = frame.to_arrow()
        yield from table.to_batches()


def _maybe_inspect(lazyframe: PolarsLazyFrame) -> None:
    inspect_fn = getattr(lazyframe, "inspect", None)
    if not callable(inspect_fn):
        return
    try:
        inspect_fn()
    except PolarsError:
        return


ColumnarStreamAdapter = RecordBatchReaderStream | LazyFrameStream


__all__ = [
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "RecordBatchReaderStream",
]
