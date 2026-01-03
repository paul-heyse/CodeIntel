"""Columnar stream protocol and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.polars_collect import (
    PolarsExecutionOptions,
    collect_batches,
    collect_lazyframe,
)
from codeintel.core.columnar.schema import unify_schema_for_batches
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.duckdb_types import DuckDBRelation

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


@runtime_checkable
class ColumnarStream(Protocol):
    """Protocol for columnar streaming sources."""

    @property
    def schema(self) -> pa.Schema:
        """Return the Arrow schema for the stream.

        Returns
        -------
        pyarrow.Schema
            Schema describing the stream output.
        """
        ...

    def to_reader(self, *, batch_size: int) -> pa.RecordBatchReader:
        """Return a RecordBatchReader for the stream.

        Parameters
        ----------
        batch_size
            Target batch size for stream readers that support it.

        Returns
        -------
        pyarrow.RecordBatchReader
            Reader over the stream batches.
        """
        ...

    def to_lazyframe(self) -> PolarsLazyFrame:
        """Return a Polars LazyFrame for the stream.

        Returns
        -------
        polars.LazyFrame
            LazyFrame view of the stream.

        Raises
        ------
        RuntimeError
            If Polars is unavailable for conversion.
        """
        ...

    def to_table(self) -> pa.Table:
        """Return a fully materialized Arrow table (last resort).

        Returns
        -------
        pyarrow.Table
            Materialized table containing the stream data.
        """
        ...


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
        try:
            dataset = ds.dataset(self.reader)
        except (ValueError, pa.ArrowInvalid):
            frame = pl.from_arrow(self.reader)
            if isinstance(frame, pl.Series):
                return frame.to_frame().lazy()
            return frame.lazy()
        return pl.scan_pyarrow_dataset(dataset)

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


def stream_from_reader(reader: pa.RecordBatchReader) -> RecordBatchReaderStream:
    """Wrap a RecordBatchReader as a ColumnarStream.

    Returns
    -------
    RecordBatchReaderStream
        Columnar stream adapter for the reader.
    """
    return RecordBatchReaderStream(reader)


def stream_from_table(
    table: pa.Table,
    *,
    batch_size: int | None = None,
) -> RecordBatchReaderStream:
    """Wrap an Arrow table as a ColumnarStream.

    Returns
    -------
    RecordBatchReaderStream
        Columnar stream adapter for the table.
    """
    batches = table.to_batches(max_chunksize=batch_size) if batch_size else table.to_batches()
    reader = pa.RecordBatchReader.from_batches(table.schema, batches)
    return RecordBatchReaderStream(reader)


def stream_from_relation(
    relation: DuckDBRelation,
    *,
    batch_size: int | None = None,
) -> RecordBatchReaderStream:
    """Wrap a DuckDB relation as a ColumnarStream.

    Returns
    -------
    RecordBatchReaderStream
        Columnar stream adapter for the relation.
    """
    reader = relation.fetch_record_batch(batch_size or DEFAULT_ARROW_BATCH_SIZE)
    return RecordBatchReaderStream(reader)


def coerce_arrow_reader(
    value: object,
    *,
    batch_size: int | None = None,
) -> pa.RecordBatchReader | None:
    """Coerce interoperability inputs into a RecordBatchReader.

    Parameters
    ----------
    value
        Candidate object implementing ``__arrow_c_stream__`` or ``__dataframe__``.
    batch_size
        Optional batch size when materializing from tables.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader when coercion succeeds, otherwise None.
    """
    if isinstance(value, pa.RecordBatchReader):
        return value
    reader = _import_c_stream(value)
    if reader is not None:
        return reader
    table = _table_from_interchange(value)
    if table is None:
        return None
    batches = table.to_batches(max_chunksize=batch_size) if batch_size else table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


def coerce_arrow_table(value: object) -> pa.Table | None:
    """Coerce interoperability inputs into an Arrow table.

    Parameters
    ----------
    value
        Candidate object implementing ``__arrow_c_stream__`` or ``__dataframe__``.

    Returns
    -------
    pyarrow.Table | None
        Table when coercion succeeds, otherwise None.
    """
    if isinstance(value, pa.Table):
        return value
    reader = _import_c_stream(value)
    if reader is not None:
        return pa.Table.from_batches(list(reader), schema=reader.schema)
    return _table_from_interchange(value)


def _import_c_stream(value: object) -> pa.RecordBatchReader | None:
    stream_fn = getattr(value, "__arrow_c_stream__", None)
    if not callable(stream_fn):
        return None
    capsule = stream_fn()
    importer = getattr(pa.RecordBatchReader, "_import_from_c", None)
    if callable(importer):
        try:
            return importer(capsule)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    return None


def _table_from_interchange(value: object) -> pa.Table | None:
    dataframe_fn = getattr(value, "__dataframe__", None)
    if not callable(dataframe_fn):
        return None
    interchange = dataframe_fn()
    module = getattr(pa, "interchange", None)
    if module is None:
        return None
    from_dataframe = getattr(module, "from_dataframe", None)
    if not callable(from_dataframe):
        return None
    kwargs: dict[str, object] = {}
    try:
        params = signature(from_dataframe).parameters
    except (TypeError, ValueError):
        params = {}
    if "allow_copy" in params:
        kwargs["allow_copy"] = False
    try:
        return from_dataframe(interchange, **kwargs)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


__all__ = [
    "ColumnarStream",
    "ColumnarStreamAdapter",
    "LazyFrameStream",
    "RecordBatchReaderStream",
    "coerce_arrow_reader",
    "coerce_arrow_table",
    "stream_from_reader",
    "stream_from_relation",
    "stream_from_table",
]
