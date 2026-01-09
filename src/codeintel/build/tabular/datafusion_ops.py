"""Optional DataFusion execution helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pyarrow as pa

try:
    import datafusion as _datafusion
except ImportError:
    _datafusion = None



@runtime_checkable
class DataFusionDataFrame(Protocol):
    """Minimal DataFusion DataFrame protocol for Arrow materialization."""

    def collect(self) -> Sequence[pa.RecordBatch] | pa.Table:
        """Collect results as record batches or a table."""
        ...

    def to_arrow_table(self) -> pa.Table:
        """Return the result as an Arrow table."""
        ...

    def schema(self) -> pa.Schema:
        """Return the DataFusion schema for the frame."""
        ...


@runtime_checkable
class DataFusionSession(Protocol):
    """Minimal DataFusion SessionContext protocol."""

    def sql(self, query: str) -> DataFusionDataFrame:
        """Run a SQL query and return the DataFusion frame."""
        ...

    def register_table(self, name: str, table: object) -> None:
        """Register a table in the session catalog."""
        ...

    def register_record_batches(self, name: str, batches: Sequence[pa.RecordBatch]) -> None:
        """Register record batches in the session catalog."""
        ...

    def from_substrait(self, plan: bytes) -> DataFusionDataFrame:
        """Build a DataFusion frame from a Substrait plan."""
        ...


def datafusion_available() -> bool:
    """Return True when the DataFusion module is available.

    Returns
    -------
    bool
        True when DataFusion can be imported.
    """
    return _datafusion is not None


def session_context() -> DataFusionSession:
    """Return a DataFusion SessionContext or raise when unavailable.

    Returns
    -------
    DataFusionSession
        Session context for DataFusion execution.

    Raises
    ------
    RuntimeError
        Raised when DataFusion or SessionContext is unavailable.
    """
    if _datafusion is None:
        msg = "datafusion is unavailable; install datafusion to enable execution."
        raise RuntimeError(msg)
    try:
        return _datafusion.SessionContext()
    except AttributeError as exc:
        msg = "datafusion.SessionContext is unavailable in this environment."
        raise RuntimeError(msg) from exc


def register_arrow_table(ctx: DataFusionSession, name: str, table: pa.Table) -> None:
    """Register an Arrow table with a DataFusion context.

    Parameters
    ----------
    ctx
        DataFusion session context.
    name
        Table name to register.
    table
        Arrow table to register.

    Raises
    ------
    RuntimeError
        Raised when the context cannot register Arrow tables.
    """
    register = getattr(ctx, "register_table", None)
    if callable(register):
        register(name, table)
        return
    register_batches = getattr(ctx, "register_record_batches", None)
    if callable(register_batches):
        register_batches(name, table.to_batches())
        return
    msg = "DataFusion context does not support registering Arrow tables."
    raise RuntimeError(msg)


def run_sql(ctx: DataFusionSession, query: str) -> pa.RecordBatchReader:
    """Execute SQL in DataFusion and return a RecordBatchReader.

    Parameters
    ----------
    ctx
        DataFusion session context.
    query
        SQL query string to execute.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for the query results.
    """
    frame = ctx.sql(query)
    return _reader_from_frame(frame)


def run_substrait_plan(
    ctx: DataFusionSession,
    plan: bytes | bytearray | memoryview,
) -> pa.RecordBatchReader:
    """Execute a Substrait plan in DataFusion and return a RecordBatchReader.

    Parameters
    ----------
    ctx
        DataFusion session context.
    plan
        Serialized Substrait plan payload.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for the plan results.

    Raises
    ------
    RuntimeError
        Raised when the context does not support Substrait execution.
    """
    builder = getattr(ctx, "from_substrait", None)
    if not callable(builder):
        msg = "DataFusion context does not support from_substrait."
        raise RuntimeError(msg)
    frame = builder(bytes(plan))
    return _reader_from_frame(frame)


def _reader_from_frame(frame: object) -> pa.RecordBatchReader:
    if isinstance(frame, pa.RecordBatchReader):
        return frame
    if isinstance(frame, pa.Table):
        return pa.RecordBatchReader.from_batches(frame.schema, frame.to_batches())
    batches = _batches_from_frame(frame)
    return _reader_from_batches(batches, frame=frame)


def _batches_from_frame(frame: object) -> Sequence[pa.RecordBatch]:
    to_arrow = getattr(frame, "to_arrow_table", None)
    if callable(to_arrow):
        table = to_arrow()
        if isinstance(table, pa.Table):
            return table.to_batches()
    collect = getattr(frame, "collect", None)
    if callable(collect):
        collected = collect()
        if isinstance(collected, pa.Table):
            return collected.to_batches()
        if isinstance(collected, Sequence):
            if all(isinstance(batch, pa.RecordBatch) for batch in collected):
                return list(collected)
    msg = "DataFusion DataFrame did not yield Arrow batches."
    raise RuntimeError(msg)


def _reader_from_batches(
    batches: Sequence[pa.RecordBatch],
    *,
    frame: object,
) -> pa.RecordBatchReader:
    if batches:
        return pa.RecordBatchReader.from_batches(batches[0].schema, batches)
    schema = _schema_from_frame(frame) or pa.schema([])
    return pa.RecordBatchReader.from_batches(schema, [])


def _schema_from_frame(frame: object) -> pa.Schema | None:
    schema_attr = getattr(frame, "schema", None)
    if callable(schema_attr):
        schema = schema_attr()
        if isinstance(schema, pa.Schema):
            return schema
        to_arrow = getattr(schema, "to_arrow_schema", None)
        if callable(to_arrow):
            arrow_schema = to_arrow()
            if isinstance(arrow_schema, pa.Schema):
                return arrow_schema
    return None


__all__ = [
    "DataFusionDataFrame",
    "DataFusionSession",
    "datafusion_available",
    "register_arrow_table",
    "run_sql",
    "run_substrait_plan",
    "session_context",
]
