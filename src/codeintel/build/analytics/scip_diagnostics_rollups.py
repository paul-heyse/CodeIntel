"""SCIP diagnostics rollup helpers for post-run analytics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.analytics.utilities.snapshot import snapshot_plan
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import materialize_plan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.query_results import coerce_int

SCIP_DIAGNOSTICS_TABLE_KEY = "core.scip_diagnostics"
SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY = "analytics.scip_diagnostics_summary"
SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY = "analytics.scip_diagnostics_by_file"
SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY = "analytics.scip_diagnostics_top_messages"

type RollupSource = Sequence[Mapping[str, object]] | pa.Table | pa.RecordBatchReader


@dataclass(frozen=True, slots=True)
class ScipDiagnosticsRollups:
    """Computed rollup rows for SCIP diagnostics outputs."""

    summary_rows: list[dict[str, object]]
    by_file_rows: list[dict[str, object]]
    top_message_rows: list[dict[str, object]]


def build_scip_diagnostics_rollups(
    *,
    repo: str,
    commit: str,
    rows: RollupSource,
) -> ScipDiagnosticsRollups:
    """Build rollup rows for SCIP diagnostics datasets.

    Returns
    -------
    ScipDiagnosticsRollups
        Rollup rows for summary, by-file, and top-message datasets.
    """
    table = _diagnostics_table(rows)
    if table is None or table.num_rows == 0:
        return ScipDiagnosticsRollups([], [], [])
    created_at = datetime.now(tz=UTC)
    return ScipDiagnosticsRollups(
        summary_rows=_aggregate_rollup_rows(
            table,
            repo=repo,
            commit=commit,
            created_at=created_at,
            group_columns=("severity", "source"),
        ),
        by_file_rows=_aggregate_rollup_rows(
            table,
            repo=repo,
            commit=commit,
            created_at=created_at,
            group_columns=("rel_path", "severity", "source"),
        ),
        top_message_rows=_aggregate_rollup_rows(
            table,
            repo=repo,
            commit=commit,
            created_at=created_at,
            group_columns=("severity", "source", "code", "message"),
        ),
    )


def _diagnostics_table(rows: RollupSource) -> pa.Table | None:
    if isinstance(rows, pa.Table):
        return rows
    if isinstance(rows, pa.RecordBatchReader):
        return reader_to_table(rows)
    if not rows:
        return None
    table, _ = table_for_rows(SCIP_DIAGNOSTICS_TABLE_KEY, rows)
    return table


def _normalized_text_expr(column: str, *, columns: set[str]) -> pc.Expression:
    if column not in columns:
        return E.scalar("unknown")
    expr = E.cast(E.field(column), "string")
    trimmed = pc.call_function("utf8_trim", [expr])
    non_empty = E.and_(E.is_valid(trimmed), trimmed != E.scalar(""))
    return pc.call_function("if_else", [non_empty, trimmed, E.scalar("unknown")])


def _aggregate_rollup_rows(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    created_at: datetime,
    group_columns: Sequence[str],
) -> list[dict[str, object]]:
    if not group_columns:
        return []
    aggregated = _aggregate_rollup_table(
        table,
        repo=repo,
        commit=commit,
        group_columns=group_columns,
    )
    rows: list[dict[str, object]] = []
    selected = [*group_columns, "diagnostic_count"]
    for row in iter_rows(aggregated, selected):
        payload: dict[str, object] = {
            "repo": repo,
            "commit": commit,
            "created_at": created_at,
        }
        for column in group_columns:
            payload[column] = _coerce_text(row.get(column))
        count_value = row.get("diagnostic_count")
        count = coerce_int(count_value, ctx="diagnostic_count") if count_value is not None else 0
        payload["diagnostic_count"] = count
        rows.append(payload)
    return rows


def _aggregate_rollup_table(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    group_columns: Sequence[str],
) -> pa.Table:
    column_names = set(table.column_names)
    plan = snapshot_plan(table, repo=repo, commit=commit)
    project = {
        column: _normalized_text_expr(column, columns=column_names) for column in group_columns
    }
    plan = plan.project(project)
    plan = plan.aggregate(
        keys=[E.field(column) for column in group_columns],
        aggregates=[(group_columns[0], "count", None, "diagnostic_count")],
    )
    return materialize_plan(plan, use_threads=True)


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
