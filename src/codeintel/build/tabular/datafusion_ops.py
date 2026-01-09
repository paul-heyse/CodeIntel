"""Optional DataFusion execution helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import pyarrow as pa

from codeintel.core.columnar.plan_ops import register_external_plan_runner

try:
    import datafusion as _datafusion
except ImportError:
    _datafusion = None

if TYPE_CHECKING:
    from codeintel.core.columnar.plan_ops import ExternalPlanRequest


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

    def register_record_batches(self, name: str, partitions: list[list[pa.RecordBatch]]) -> None:
        """Register record batches in the session catalog."""
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
        register_batches(name, [table.to_batches()])
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
    TypeError
        Raised when the context does not support Substrait execution.
    """
    builder = getattr(ctx, "from_substrait", None)
    if not callable(builder):
        msg = "DataFusion context does not support from_substrait."
        raise TypeError(msg)
    frame = builder(bytes(plan))
    return _reader_from_frame(frame)


def datafusion_plan_runner(
    *,
    request: ExternalPlanRequest,
) -> pa.RecordBatchReader:
    """Execute a DataFusion plan via ExternalPlanSpec.

    Returns
    -------
    pyarrow.RecordBatchReader
        Record batch reader for plan results.

    Raises
    ------
    TypeError
        Raised when the payload is not SQL text or Substrait bytes.
    """
    _ = (
        request.dataset,
        request.filter_expr,
        request.columns,
        request.scan_options,
        request.use_threads,
    )
    ctx = session_context()
    payload = request.spec.payload
    if isinstance(payload, str):
        return run_sql(ctx, payload)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return run_substrait_plan(ctx, payload)
    if isinstance(payload, Mapping):
        sql = payload.get("sql")
        if isinstance(sql, str):
            return run_sql(ctx, sql)
        plan = payload.get("plan")
        if isinstance(plan, (bytes, bytearray, memoryview)):
            return run_substrait_plan(ctx, plan)
    msg = "DataFusion payload must be SQL text or Substrait bytes."
    raise TypeError(msg)


def register_datafusion_plan_runner(name: str = "datafusion") -> None:
    """Register the DataFusion external plan runner."""
    register_external_plan_runner(name, datafusion_plan_runner)


def _reader_from_frame(
    frame: DataFusionDataFrame | pa.RecordBatchReader | pa.Table,
) -> pa.RecordBatchReader:
    if isinstance(frame, pa.RecordBatchReader):
        return frame
    if isinstance(frame, pa.Table):
        table = cast("pa.Table", frame)
        return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    batches = _batches_from_frame(frame)
    return _reader_from_batches(batches, frame=frame)


def _batches_from_frame(frame: DataFusionDataFrame) -> Sequence[pa.RecordBatch]:
    to_arrow = getattr(frame, "to_arrow_table", None)
    if callable(to_arrow):
        table = to_arrow()
        if isinstance(table, pa.Table):
            resolved = cast("pa.Table", table)
            return resolved.to_batches()
    collect = getattr(frame, "collect", None)
    if callable(collect):
        collected = collect()
        if isinstance(collected, pa.Table):
            resolved = cast("pa.Table", collected)
            return resolved.to_batches()
        if isinstance(collected, Sequence) and all(
            isinstance(batch, pa.RecordBatch) for batch in collected
        ):
            return list(collected)
    msg = "DataFusion DataFrame did not yield Arrow batches."
    raise RuntimeError(msg)


def _reader_from_batches(
    batches: Sequence[pa.RecordBatch],
    *,
    frame: DataFusionDataFrame | pa.RecordBatchReader | pa.Table,
) -> pa.RecordBatchReader:
    if batches:
        return pa.RecordBatchReader.from_batches(batches[0].schema, batches)
    schema = _schema_from_frame(frame) or pa.schema([])
    return pa.RecordBatchReader.from_batches(schema, [])


def _schema_from_frame(
    frame: DataFusionDataFrame | pa.RecordBatchReader | pa.Table,
) -> pa.Schema | None:
    if isinstance(frame, pa.Table):
        return frame.schema
    if isinstance(frame, pa.RecordBatchReader):
        return frame.schema
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
    "datafusion_plan_runner",
    "register_arrow_table",
    "register_datafusion_plan_runner",
    "run_sql",
    "run_substrait_plan",
    "session_context",
]
