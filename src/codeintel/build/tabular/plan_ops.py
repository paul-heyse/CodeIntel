"""Acero plan helper re-exports for build pipelines."""

from __future__ import annotations

from codeintel.build.tabular.datafusion_ops import register_datafusion_plan_runner
from codeintel.build.tabular.substrait_ops import register_substrait_plan_runner
from codeintel.core.columnar.plan_ops import (
    ExternalPlanRequest,
    ExternalPlanSpec,
    HashJoinSpec,
    JoinType,
    Plan,
    QueryPlanOptions,
    ScanPlanOptions,
    build_query_plan,
    build_scan_plan,
    list_external_plan_runners,
    materialize_plan,
    register_external_plan_runner,
    run_external_plan,
)


def register_default_external_plan_runners() -> None:
    """Register optional external plan runners for build pipelines."""
    register_substrait_plan_runner()
    register_datafusion_plan_runner()


register_default_external_plan_runners()

__all__ = [
    "ExternalPlanRequest",
    "ExternalPlanSpec",
    "HashJoinSpec",
    "JoinType",
    "Plan",
    "QueryPlanOptions",
    "ScanPlanOptions",
    "build_query_plan",
    "build_scan_plan",
    "list_external_plan_runners",
    "materialize_plan",
    "register_default_external_plan_runners",
    "register_external_plan_runner",
    "run_external_plan",
]
