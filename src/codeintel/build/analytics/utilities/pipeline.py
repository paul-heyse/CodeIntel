"""Shared pipeline runner for analytics QuerySpec execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.build.analytics.utilities.finalize import finalize_analytics_reader
from codeintel.build.tabular.plan_ops import Plan, build_query_plan_for_context
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.queryspec import PROVENANCE_FIELDS, QuerySpec

if TYPE_CHECKING:
    from codeintel.build.tabular.finalize_ops import FinalizeResult

QuerySource = ds.Dataset | pa.Table


def run_analytics_pipeline(
    *,
    source: QuerySource,
    spec: QuerySpec,
    table_key: str,
    ctx: ExecutionContext,
) -> FinalizeResult:
    """Execute a QuerySpec and finalize results for analytics outputs.

    Returns
    -------
    FinalizeResult
        Finalize artifacts for the table key.
    """
    if isinstance(source, ds.Dataset):
        plan = build_query_plan_for_context(source, spec=spec, ctx=ctx)
    else:
        plan = _plan_for_table(source, spec=spec, ctx=ctx)
    reader = plan.to_reader(use_threads=ctx.resolve_use_threads())
    return finalize_analytics_reader(table_key, reader)


def _plan_for_table(
    table: pa.Table,
    *,
    spec: QuerySpec,
    ctx: ExecutionContext,
) -> Plan:
    plan = Plan.table(table)
    if spec.predicate is not None:
        plan = plan.filter(spec.predicate)
    projection = spec.project_expressions(provenance=_include_provenance(table, ctx=ctx))
    if projection:
        plan = plan.project(projection)
    return plan


def _include_provenance(table: pa.Table, *, ctx: ExecutionContext) -> bool:
    if not ctx.provenance:
        return False
    column_names = set(table.column_names)
    return all(output_name in column_names for output_name, _source_name in PROVENANCE_FIELDS)


__all__ = ["QuerySource", "run_analytics_pipeline"]
