"""SCIP diagnostics rollup helpers for post-run analytics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.analytics.compute.row_builders import buffer_for_table
from codeintel.build.analytics.utilities.snapshot import SnapshotContext, snapshot_plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.plan_builder import build_grouped_rollup_plan
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    columnar_batch_collector_for_table_key,
    table_for_rows,
)
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.query_results import coerce_int

SCIP_DIAGNOSTICS_TABLE_KEY = "core.scip_diagnostics"
SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY = "analytics.scip_diagnostics_summary"
SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY = "analytics.scip_diagnostics_by_file"
SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY = "analytics.scip_diagnostics_top_messages"

type RollupSource = (
    Sequence[Mapping[str, object]] | ColumnarRowBuffer | pa.Table | pa.RecordBatchReader
)


@dataclass(frozen=True, slots=True)
class ScipDiagnosticsRollups:
    """Computed rollup rows for SCIP diagnostics outputs."""

    summary_rows: ColumnarRowBuffer
    by_file_rows: ColumnarRowBuffer
    top_message_rows: ColumnarRowBuffer


@dataclass(frozen=True, slots=True)
class RollupSpec:
    """Rollup specification for diagnostics aggregation."""

    repo: str
    commit: str
    created_at: datetime
    table_key: str
    group_columns: Sequence[str]
    ctx: ExecutionContext | RuntimeExecutionContext | None


def build_scip_diagnostics_rollups(
    *,
    repo: str,
    commit: str,
    rows: RollupSource,
    ctx: ExecutionContext | RuntimeExecutionContext | None = None,
) -> ScipDiagnosticsRollups:
    """Build rollup rows for SCIP diagnostics datasets.

    Returns
    -------
    ScipDiagnosticsRollups
        Rollup rows for summary, by-file, and top-message datasets.
    """
    table = _diagnostics_table(rows)
    if table is None or table.num_rows == 0:
        return ScipDiagnosticsRollups(
            summary_rows=buffer_for_table(SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY),
            by_file_rows=buffer_for_table(SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY),
            top_message_rows=buffer_for_table(SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY),
        )
    created_at = datetime.now(tz=UTC)
    return ScipDiagnosticsRollups(
        summary_rows=_aggregate_rollup_rows(
            table,
            RollupSpec(
                repo=repo,
                commit=commit,
                created_at=created_at,
                table_key=SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY,
                group_columns=("severity", "source"),
                ctx=ctx,
            ),
        ),
        by_file_rows=_aggregate_rollup_rows(
            table,
            RollupSpec(
                repo=repo,
                commit=commit,
                created_at=created_at,
                table_key=SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY,
                group_columns=("rel_path", "severity", "source"),
                ctx=ctx,
            ),
        ),
        top_message_rows=_aggregate_rollup_rows(
            table,
            RollupSpec(
                repo=repo,
                commit=commit,
                created_at=created_at,
                table_key=SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY,
                group_columns=("severity", "source", "code", "message"),
                ctx=ctx,
            ),
        ),
    )


def _diagnostics_table(rows: RollupSource) -> pa.Table | None:
    if isinstance(rows, pa.Table):
        return rows
    if isinstance(rows, ColumnarRowBuffer):
        collector = columnar_batch_collector_for_table_key(SCIP_DIAGNOSTICS_TABLE_KEY)
        collector.extend(rows)
        return reader_to_table(collector.to_reader())
    if isinstance(rows, pa.RecordBatchReader):
        return reader_to_table(rows)
    if not rows:
        return None
    table, _ = table_for_rows(SCIP_DIAGNOSTICS_TABLE_KEY, rows)
    return table


def _normalized_text_expr(column: str, *, columns: set[str]) -> Expression:
    if column not in columns:
        return E.scalar("unknown")
    expr = E.cast(E.field(column), "string")
    trimmed = E.utf8_trim(expr)
    non_empty = E.and_(E.is_valid(trimmed), trimmed != E.scalar(""))
    return E.if_else(non_empty, trimmed, E.scalar("unknown"))


def _aggregate_rollup_rows(table: pa.Table, spec: RollupSpec) -> ColumnarRowBuffer:
    if not spec.group_columns:
        return buffer_for_table(spec.table_key)
    aggregated = _aggregate_rollup_table(
        table,
        repo=spec.repo,
        commit=spec.commit,
        group_columns=spec.group_columns,
        ctx=spec.ctx,
    )
    buffer = buffer_for_table(spec.table_key)
    selected = [*spec.group_columns, "diagnostic_count"]
    for row in iter_rows(aggregated, selected):
        payload: dict[str, object] = {
            "repo": spec.repo,
            "commit": spec.commit,
            "created_at": spec.created_at,
        }
        for column in spec.group_columns:
            payload[column] = _coerce_text(row.get(column))
        count_value = row.get("diagnostic_count")
        count = coerce_int(count_value, ctx="diagnostic_count") if count_value is not None else 0
        payload["diagnostic_count"] = count
        buffer.append(payload)
    return buffer


def _aggregate_rollup_table(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    group_columns: Sequence[str],
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    column_names = set(table.column_names)
    plan = snapshot_plan(
        table,
        context=SnapshotContext(repo=repo, commit=commit, ctx=ctx),
    )
    project = {
        column: _normalized_text_expr(column, columns=column_names) for column in group_columns
    }
    plan = plan.project(project)
    plan = build_grouped_rollup_plan(
        plan,
        keys=group_columns,
        aggregates=[(group_columns[0], "count", None, "diagnostic_count")],
    )
    execution_ctx = resolve_execution_context(resolve_columnar_context(ctx))
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


def _coerce_text(value: object | None, *, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


__all__ = [
    "SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY",
    "SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY",
    "SCIP_DIAGNOSTICS_TABLE_KEY",
    "SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY",
    "ScipDiagnosticsRollups",
    "build_scip_diagnostics_rollups",
]
