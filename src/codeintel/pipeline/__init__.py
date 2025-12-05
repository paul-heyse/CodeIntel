"""Pipeline orchestration and export utilities for batch CodeIntel operations.

This package provides:

- **Declarative specs**: Define pipelines as ordered stages via :class:`PipelineSpec`
- **Planning**: Build execution plans via :func:`build_pipeline_plan`
- **Execution**: Run pipelines via :func:`run_pipeline`
- **Operation planning**: Compute prereqs via :func:`build_pipeline_for_operation`
- **Run tracking**: Access run/step records via :mod:`codeintel.pipeline.run_registry`

Quick Start
-----------
>>> from codeintel.pipeline import FULL_PIPELINE, run_pipeline
>>> # result = run_pipeline(
>>> #     spec=FULL_PIPELINE,
>>> #     options=PipelinePlanOptions(
>>> #         snapshot=...,
>>> #         paths=...,
>>> #         gateway=...,
>>> #         tools=...,
>>> #     ),
>>> # )

Operation-Driven Orchestration
-----------------------------
>>> from codeintel.pipeline import build_pipeline_for_operation
>>> # spec = build_pipeline_for_operation("function.summary", snapshot)
>>> # assert spec.id == "full"  # Requires graphs, so needs full pipeline
"""

from codeintel.pipeline.execution import tracking as run_registry
from codeintel.pipeline.execution.runner import run_pipeline
from codeintel.pipeline.planning.op_planner import (
    OperationPrereqOptions,
    OpPrereqSummary,
    build_pipeline_for_operation,
    build_prereq_summary,
    ensure_prerequisites_for_operation,
)
from codeintel.pipeline.planning.planner import (
    AnalyticsStagePlan,
    GraphsStagePlan,
    IngestionStagePlan,
    PipelinePlan,
    PipelinePlanOptions,
    build_pipeline_plan,
)
from codeintel.pipeline.spec.model import (
    ANALYTICS_ONLY,
    FULL_PIPELINE,
    GRAPHS_ONLY,
    INGEST_ONLY,
    NOOP_PIPELINE,
    PIPELINE_SPECS,
    PipelineSpec,
    PipelineStage,
    StageModule,
    get_pipeline_spec,
    list_pipeline_specs,
)

__all__ = [
    "ANALYTICS_ONLY",
    "FULL_PIPELINE",
    "GRAPHS_ONLY",
    "INGEST_ONLY",
    "NOOP_PIPELINE",
    "PIPELINE_SPECS",
    "AnalyticsStagePlan",
    "GraphsStagePlan",
    "IngestionStagePlan",
    "OpPrereqSummary",
    "OperationPrereqOptions",
    "PipelinePlan",
    "PipelinePlanOptions",
    "PipelineSpec",
    "PipelineStage",
    "StageModule",
    "build_pipeline_for_operation",
    "build_pipeline_plan",
    "build_prereq_summary",
    "ensure_prerequisites_for_operation",
    "get_pipeline_spec",
    "list_pipeline_specs",
    "run_pipeline",
    "run_registry",
]
