"""Normalize and convert tabular inputs for columnar workflows."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Protocol, TypedDict, cast, runtime_checkable

import polars as pl
import pyarrow as pa
from polars.exceptions import PolarsError
from pyiceberg.table import DataScan

from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBRelation

type PolarsDataFrame = pl.DataFrame
type PolarsLazyFrame = pl.LazyFrame
type IcebergDataScan = DataScan


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
        """Return a fully materialized Arrow table.

        Returns
        -------
        pyarrow.Table
            Materialized table containing the stream data.
        """
        ...


type TabularRelation = DuckDBRelation
type TabularFrame = PolarsLazyFrame
type TabularInput = (
    TabularRelation
    | pa.RecordBatchReader
    | pa.Table
    | TabularFrame
    | ColumnarStream
    | IcebergDataScan
    | object
)


class CollectKwargs(TypedDict, total=False):
    engine: str
    streaming: bool
    optimization_flags: object
    query_opt_flags: object
    optimizations: object


class CollectBatchesKwargs(CollectKwargs, total=False):
    batch_size: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class PolarsExecutionOptions:
    """Execution options for Polars LazyFrame collection."""

    streaming: bool = True
    query_opt_flags: object | None = None
    inspect: bool = False
    streaming_fallback: bool = True


def _is_iceberg_scan(value: object) -> bool:
    return isinstance(value, DataScan)


def _iceberg_reader(
    scan: IcebergDataScan,
    *,
    batch_size: int | None = None,
) -> pa.RecordBatchReader:
    _ = batch_size
    return scan.to_arrow_batch_reader()


def _iceberg_lazyframe(scan: IcebergDataScan) -> PolarsLazyFrame:
    frame = scan.to_polars()
    if isinstance(frame, pl.DataFrame):
        return frame.lazy()
    return cast("PolarsLazyFrame", frame)


def _iceberg_table(scan: IcebergDataScan) -> pa.Table:
    return scan.to_arrow()


def collect_lazyframe(
    lazyframe: PolarsLazyFrame,
    *,
    options: PolarsExecutionOptions,
) -> PolarsDataFrame:
    """Collect a LazyFrame with typed option handling.

    Returns
    -------
    polars.DataFrame
        Collected DataFrame.
    """
    collect_target = cast("Callable[..., object]", lazyframe.collect)
    kwargs = _collect_kwargs(collect_target, options=options)
    collect_fn = _collect_callable(lazyframe.collect)
    return collect_fn(**kwargs)


def collect_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    options: PolarsExecutionOptions,
) -> Sequence[PolarsDataFrame]:
    """Collect a LazyFrame into batches with typed option handling.

    Returns
    -------
    Sequence[polars.DataFrame]
        Collected DataFrame batches.
    """
    collect_target = cast("Callable[..., object]", lazyframe.collect_batches)
    kwargs = _collect_batches_kwargs(
        collect_target,
        batch_size=batch_size,
        options=options,
    )
    collect_fn = _collect_batches_callable(lazyframe.collect_batches)
    return collect_fn(**kwargs)


def _collect_kwargs(
    func: Callable[..., object],
    *,
    options: PolarsExecutionOptions,
) -> CollectKwargs:
    params = _param_names(func)
    kwargs: CollectKwargs = {}
    _populate_streaming_kwargs(kwargs, params, streaming=options.streaming)
    _populate_opt_flags(kwargs, params, options.query_opt_flags)
    return kwargs


def _collect_batches_kwargs(
    func: Callable[..., object],
    *,
    batch_size: int,
    options: PolarsExecutionOptions,
) -> CollectBatchesKwargs:
    params = _param_names(func)
    kwargs: CollectBatchesKwargs = {}
    if "chunk_size" in params:
        kwargs["chunk_size"] = batch_size
    elif "batch_size" in params:
        kwargs["batch_size"] = batch_size
    _populate_streaming_kwargs(kwargs, params, streaming=options.streaming)
    _populate_opt_flags(kwargs, params, options.query_opt_flags)
    return kwargs


def _populate_streaming_kwargs(
    kwargs: CollectKwargs,
    params: frozenset[str],
    *,
    streaming: bool,
) -> None:
    if "engine" in params:
        if streaming:
            kwargs["engine"] = "streaming"
        return
    if "streaming" in params:
        kwargs["streaming"] = streaming


def _populate_opt_flags(
    kwargs: CollectKwargs,
    params: frozenset[str],
    query_opt_flags: object | None,
) -> None:
    if query_opt_flags is None:
        return
    if "optimization_flags" in params:
        kwargs["optimization_flags"] = query_opt_flags
    elif "query_opt_flags" in params:
        kwargs["query_opt_flags"] = query_opt_flags
    elif "optimizations" in params:
        kwargs["optimizations"] = query_opt_flags


def _param_names(func: Callable[..., object]) -> frozenset[str]:
    try:
        params = signature(func).parameters
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(params)


def _collect_callable(
    func: object,
) -> Callable[..., PolarsDataFrame]:
    return cast("Callable[..., PolarsDataFrame]", func)


def _collect_batches_callable(
    func: object,
) -> Callable[..., Sequence[PolarsDataFrame]]:
    return cast("Callable[..., Sequence[PolarsDataFrame]]", func)


def to_record_batch_reader(
    value: TabularInput,
    *,
    batch_size: int,
    options: PolarsExecutionOptions | None = None,
) -> pa.RecordBatchReader:
    """Convert a tabular input into a RecordBatchReader.

    Parameters
    ----------
    value
        Tabular input to normalize.
    batch_size
        Target batch size when streaming LazyFrame or table inputs.
    options
        Optional execution options for Polars LazyFrame collection.

    Returns
    -------
    pyarrow.RecordBatchReader
        Arrow reader for the input.

    Raises
    ------
    TypeError
        If the value cannot be coerced into a reader.
    ValueError
        If batch_size is not positive.
    """
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    reader: pa.RecordBatchReader | None = None
    if isinstance(value, pa.RecordBatchReader):
        reader = value
    elif _is_iceberg_scan(value):
        reader = _iceberg_reader(cast("IcebergDataScan", value), batch_size=batch_size)
    elif isinstance(value, ColumnarStream):
        reader = value.to_reader(batch_size=batch_size)
    elif isinstance(value, pa.Table):
        table = cast("pa.Table", value)
        batches = table.to_batches(max_chunksize=batch_size)
        reader = pa.RecordBatchReader.from_batches(table.schema, batches)
    elif isinstance(value, DuckDBRelation):
        reader = value.fetch_arrow_reader()
    elif isinstance(value, pl.LazyFrame):
        reader = _lazyframe_to_reader(value, batch_size=batch_size, options=options)
    if reader is None:
        reader = coerce_arrow_reader(value, batch_size=batch_size)
    if reader is None:
        msg = f"Unsupported tabular input: {type(value)!r}"
        raise TypeError(msg)
    return reader


def to_table(
    value: TabularInput,
    *,
    batch_size: int,
    options: PolarsExecutionOptions | None = None,
) -> pa.Table:
    """Convert a tabular input into an Arrow table.

    Parameters
    ----------
    value
        Tabular input to convert.
    batch_size
        Target batch size for streaming inputs.
    options
        Optional execution options for Polars LazyFrame collection.

    Returns
    -------
    pyarrow.Table
        Arrow table representation of the input.
    """
    if isinstance(value, pa.Table):
        return value
    if _is_iceberg_scan(value):
        return _iceberg_table(cast("IcebergDataScan", value))
    if isinstance(value, DuckDBRelation):
        reader = value.fetch_arrow_reader()
        return pa.Table.from_batches(list(reader), schema=reader.schema)
    if isinstance(value, ColumnarStream):
        return value.to_table()
    if isinstance(value, pl.LazyFrame):
        frame = collect_lazyframe(value, options=_resolve_polars_options(options))
        return frame.to_arrow()
    reader = to_record_batch_reader(value, batch_size=batch_size, options=options)
    return pa.Table.from_batches(list(reader), schema=reader.schema)


def _lazyframe_from_known_types(value: TabularInput) -> PolarsLazyFrame | None:
    if isinstance(value, pl.LazyFrame):
        lazyframe: PolarsLazyFrame | None = value
    elif _is_iceberg_scan(value):
        lazyframe = _iceberg_lazyframe(cast("IcebergDataScan", value))
    elif isinstance(value, ColumnarStream):
        lazyframe = value.to_lazyframe()
    elif isinstance(value, pa.Table):
        lazyframe = _table_to_lazyframe(value)
    elif isinstance(value, pa.RecordBatchReader):
        lazyframe = _arrow_reader_to_lazyframe(value)
    elif isinstance(value, DuckDBRelation):
        lazyframe = _arrow_reader_to_lazyframe(value.fetch_arrow_reader())
    else:
        lazyframe = None
    return lazyframe


def _lazyframe_from_interchange(value: object) -> PolarsLazyFrame | None:
    reader = coerce_arrow_reader(value, batch_size=None)
    if reader is not None:
        return _arrow_reader_to_lazyframe(reader)
    table = coerce_arrow_table(value)
    if table is not None:
        return _table_to_lazyframe(table)
    return None


def to_lazyframe(value: TabularInput) -> PolarsLazyFrame:
    """Convert a tabular input into a Polars LazyFrame.

    Parameters
    ----------
    value
        Tabular input to convert.

    Returns
    -------
    polars.LazyFrame
        LazyFrame representing the input data.

    Raises
    ------
    TypeError
        If the value cannot be coerced into a LazyFrame.
    """
    lazyframe = _lazyframe_from_known_types(value)
    if lazyframe is None:
        lazyframe = _lazyframe_from_interchange(value)
    if lazyframe is None:
        msg = f"Unsupported tabular input for LazyFrame: {type(value)!r}"
        raise TypeError(msg)
    return lazyframe


def to_relation(
    conn: DuckDBConnection,
    value: TabularInput,
    *,
    name_hint: str | None = None,
) -> DuckDBRelation:
    """Coerce a tabular input into a DuckDB relation.

    Parameters
    ----------
    conn
        DuckDB connection used for registration.
    value
        Tabular input to register.
    name_hint
        Optional prefix for the registered name.

    Returns
    -------
    duckdb.DuckDBPyRelation
        DuckDB relation for the provided input.
    """
    if isinstance(value, DuckDBRelation):
        return value
    name = register_ephemeral(conn, value, prefix=name_hint or "tmp")
    return conn.table(name)


def register_ephemeral(
    conn: DuckDBConnection,
    obj: TabularInput,
    *,
    prefix: str = "tmp",
) -> str:
    """Register a tabular object under a unique ephemeral name.

    Parameters
    ----------
    conn
        DuckDB connection used for registration.
    obj
        Tabular object to register.
    prefix
        Name prefix used for the generated registration name.

    Returns
    -------
    str
        Name registered in DuckDB for the object.
    """
    safe_prefix = _sanitize_name(prefix)
    name = f"{safe_prefix}_{uuid.uuid4().hex}"
    if _is_iceberg_scan(obj):
        obj = _iceberg_reader(
            cast("IcebergDataScan", obj),
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        )
    elif isinstance(obj, ColumnarStream):
        obj = obj.to_reader(batch_size=DEFAULT_ARROW_BATCH_SIZE)
    conn.register(name, obj)
    return name


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
    if _is_iceberg_scan(value):
        return _iceberg_reader(cast("IcebergDataScan", value), batch_size=batch_size)
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
    if _is_iceberg_scan(value):
        return _iceberg_table(cast("IcebergDataScan", value))
    reader = _import_c_stream(value)
    if reader is not None:
        return pa.Table.from_batches(list(reader), schema=reader.schema)
    return _table_from_interchange(value)


def _lazyframe_to_reader(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    options: PolarsExecutionOptions | None,
) -> pa.RecordBatchReader:
    resolved = _resolve_polars_options(options)

    def _iter_batches() -> Iterator[pa.RecordBatch]:
        streaming = resolved.streaming
        try:
            yield from _iter_polars_batches(
                lazyframe,
                batch_size=batch_size,
                options=resolved,
            )
        except PolarsError:
            if streaming and resolved.streaming_fallback:
                fallback = PolarsExecutionOptions(
                    streaming=False,
                    query_opt_flags=resolved.query_opt_flags,
                    inspect=resolved.inspect,
                    streaming_fallback=resolved.streaming_fallback,
                )
                yield from _iter_polars_batches(
                    lazyframe,
                    batch_size=batch_size,
                    options=fallback,
                )
            else:
                raise

    schema = lazyframe.collect_schema().to_arrow()
    return pa.RecordBatchReader.from_batches(schema, _iter_batches())


def _iter_polars_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    options: PolarsExecutionOptions,
) -> Iterator[pa.RecordBatch]:
    if options.inspect:
        _maybe_inspect(lazyframe)
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


def _resolve_polars_options(options: PolarsExecutionOptions | None) -> PolarsExecutionOptions:
    return options if options is not None else PolarsExecutionOptions()


def _arrow_reader_to_lazyframe(reader: pa.RecordBatchReader) -> PolarsLazyFrame:
    frame = pl.from_arrow(reader)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


def _table_to_lazyframe(table: pa.Table) -> PolarsLazyFrame:
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        return frame.to_frame().lazy()
    return frame.lazy()


_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")


def _sanitize_name(prefix: str) -> str:
    cleaned = _NAME_SANITIZER.sub("_", prefix.strip())
    return cleaned if cleaned else "tmp"


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
    "PolarsExecutionOptions",
    "TabularFrame",
    "TabularInput",
    "TabularRelation",
    "coerce_arrow_reader",
    "coerce_arrow_table",
    "collect_batches",
    "collect_lazyframe",
    "register_ephemeral",
    "to_lazyframe",
    "to_record_batch_reader",
    "to_relation",
    "to_table",
]
