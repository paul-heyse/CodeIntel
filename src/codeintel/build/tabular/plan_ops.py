"""Acero plan helper re-exports for build pipelines."""

from __future__ import annotations

from codeintel.core.columnar.external_plans import register_default_external_plan_runners
from codeintel.core.columnar.plan_ops import (
    ExternalPlanRequest,
    ExternalPlanSpec,
    HashJoinSpec,
    JoinType,
    Plan,
    QueryPlanOptions,
    ScanPlanOptions,
    build_query_plan,
    build_query_plan_for_context,
    build_scan_plan,
    list_external_plan_runners,
    materialize_plan,
    query_plan_options_for_context,
    register_external_plan_runner,
    run_external_plan,
)

__all__ = [
    "ExternalPlanRequest",
    "ExternalPlanSpec",
    "HashJoinSpec",
    "JoinType",
    "Plan",
    "QueryPlanOptions",
    "ScanPlanOptions",
    "build_query_plan",
    "build_query_plan_for_context",
    "build_scan_plan",
    "list_external_plan_runners",
    "materialize_plan",
    "query_plan_options_for_context",
    "register_default_external_plan_runners",
    "register_external_plan_runner",
    "run_external_plan",
]
