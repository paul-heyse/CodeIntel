"""Parity checks between Arrow plan specs and QuerySpec scan plans."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.plan_ops import Plan, build_query_plan
from codeintel.serving.semantic.arrow_plan_builder import ArrowPlanSpec, build_arrow_query_spec


def _apply_arrow_plan(table: pa.Table, plan_spec: ArrowPlanSpec) -> pa.Table:
    plan = Plan.table(table)
    if plan_spec.filter_expr is not None:
        plan = plan.filter(plan_spec.filter_expr)
    if plan_spec.order_by:
        plan = plan.order_by(sort_keys=plan_spec.order_by)
    if plan_spec.projections:
        plan = plan.project(plan_spec.projections)
    return plan.to_table(use_threads=False)


def test_arrow_query_spec_scan_plan_parity() -> None:
    """QuerySpec scan plans should match Arrow plan execution."""
    table = pa.table(
        {
            "id": [1, 2, 3],
            "kind": ["call", "def", "call"],
        }
    )
    dataset = ds.dataset(table)
    plan_spec = ArrowPlanSpec(
        filter_expr=E.field("kind") == E.scalar("call"),
        projections={"id": E.field("id"), "kind": E.field("kind")},
        order_by=(),
        limit=None,
    )
    query_spec = build_arrow_query_spec(plan_spec)
    scan_plan = build_query_plan(dataset, spec=query_spec)
    scanned = scan_plan.to_table(use_threads=False)
    expected = _apply_arrow_plan(table, plan_spec)
    assert list(iter_rows(scanned)) == list(iter_rows(expected))
