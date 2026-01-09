"""Snapshot-scoped Plan helpers for analytics utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import Expression
from codeintel.core.columnar.plan_builder import (
    SchemaPlanDefaultsRequest,
    plan_from_schema_defaults,
    require_columns,
)
from codeintel.core.columnar.plan_builder import (
    build_snapshot_plan as _build_snapshot_plan,
)
from codeintel.core.columnar.plan_builder import (
    build_snapshot_query_spec as _build_snapshot_query_spec,
)
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.queryspec import QuerySpec
from codeintel.core.columnar.schema_metadata import decode_metadata
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.schemas.service import get_schema_service


@dataclass(frozen=True, slots=True)
class SnapshotContext:
    """Snapshot context for analytics plan helpers."""

    repo: str | None = None
    commit: str | None = None
    ctx: ExecutionContext | RuntimeExecutionContext | None = None
    table_key: str | None = None


def build_snapshot_query_spec(
    *,
    base_cols: Sequence[str],
    context: SnapshotContext | None = None,
    computed: Sequence[tuple[str, Expression]] = (),
    table: pa.Table | None = None,
) -> QuerySpec:
    """Build a QuerySpec scoped to a repo/commit snapshot.

    Parameters
    ----------
    base_cols
        Base columns to include in the projection.
    context
        Snapshot context containing repo/commit scope and defaults.
    computed
        Computed projection expressions.
    table
        Optional table used to validate column availability.

    Returns
    -------
    QuerySpec
        Snapshot-scoped query specification with optional projection.
    """
    resolved_context = context or SnapshotContext()
    return _build_snapshot_query_spec(
        base_cols=base_cols,
        repo=resolved_context.repo,
        commit=resolved_context.commit,
        computed=computed,
        table=table,
    )


def snapshot_plan(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
    context: SnapshotContext | None = None,
) -> Plan:
    """Build a Plan scoped to a repo/commit snapshot.

    Parameters
    ----------
    table
        Input table to scope.
    columns
        Optional column projection to apply after filtering.
    context
        Snapshot context containing repo/commit scope and defaults.

    Returns
    -------
    Plan
        Plan filtered to the snapshot and optionally projected.
    """
    base_cols = tuple(columns or ())
    resolved_context = context or SnapshotContext()
    spec = build_snapshot_query_spec(
        base_cols=base_cols,
        context=resolved_context,
        table=table,
    )
    resolved_ctx = resolve_columnar_context(resolved_context.ctx)
    resolved_table_key = resolved_context.table_key
    if resolved_table_key is None:
        metadata = decode_metadata(table.schema.metadata)
        meta_value = metadata.get("codeintel.table_key")
        if isinstance(meta_value, str):
            resolved_table_key = meta_value
    if resolved_table_key is not None:
        dataset = ds.dataset(table)
        return plan_from_schema_defaults(
            schema_service=get_schema_service(),
            request=SchemaPlanDefaultsRequest(
                table_key=resolved_table_key,
                dataset=dataset,
                predicate=spec.predicate,
                columns=spec.scan_columns(provenance=False),
                ctx=resolved_ctx,
            ),
        )
    return _build_snapshot_plan(table=table, spec=spec, ctx=resolved_ctx)


def snapshot_table(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
    context: SnapshotContext | None = None,
) -> pa.Table:
    """Materialize a snapshot-scoped Plan.

    Returns
    -------
    pyarrow.Table
        Snapshot-scoped table.
    """
    plan = snapshot_plan(
        table,
        columns=columns,
        context=context,
    )
    resolved_context = context or SnapshotContext()
    execution_ctx = resolve_execution_context(resolve_columnar_context(resolved_context.ctx))
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


def snapshot_reader(
    table: pa.Table,
    *,
    columns: Sequence[str] | None = None,
    context: SnapshotContext | None = None,
) -> pa.RecordBatchReader:
    """Materialize a snapshot-scoped Plan as a reader.

    Returns
    -------
    pyarrow.RecordBatchReader
        Snapshot-scoped reader.
    """
    plan = snapshot_plan(
        table,
        columns=columns,
        context=context,
    )
    resolved_context = context or SnapshotContext()
    execution_ctx = resolve_execution_context(resolve_columnar_context(resolved_context.ctx))
    return ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)


__all__ = [
    "SnapshotContext",
    "build_snapshot_query_spec",
    "require_columns",
    "snapshot_plan",
    "snapshot_reader",
    "snapshot_table",
]
