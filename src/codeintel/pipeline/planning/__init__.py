"""Pipeline planning and prerequisite resolution.

This package contains planning utilities:

- build_pipeline_plan: Build an execution plan from a spec
- PipelinePlan: The execution plan data structure
- ensure_prerequisites_for_operation: Resolve and execute prerequisites for an operation
- build_pipeline_for_operation: Build a pipeline spec for an operation
"""

from __future__ import annotations

from codeintel.pipeline.planning.op_planner import (
    OperationPrereqOptions,
    OpPrereqSummary,
    build_pipeline_for_operation,
    build_prereq_summary,
    ensure_prerequisites_for_operation,
    get_required_table_keys_for_operation,
)
from codeintel.pipeline.planning.planner import (
    PipelinePlan,
    PipelinePlanOptions,
    build_pipeline_plan,
)

__all__ = [
    "OpPrereqSummary",
    "OperationPrereqOptions",
    "PipelinePlan",
    "PipelinePlanOptions",
    "build_pipeline_for_operation",
    "build_pipeline_plan",
    "build_prereq_summary",
    "ensure_prerequisites_for_operation",
    "get_required_table_keys_for_operation",
]
