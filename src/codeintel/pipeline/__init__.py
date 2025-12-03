"""Pipeline orchestration and export utilities for batch CodeIntel operations.

This package provides:

- **Declarative specs**: Define pipelines as ordered stages via :class:`PipelineSpec`
- **Planning**: Build execution plans via :func:`build_pipeline_plan`
- **Execution**: Run pipelines via :func:`run_pipeline`
- **Run tracking**: Access run/step records via :mod:`codeintel.pipeline.run_registry`

Quick Start
-----------
>>> from codeintel.pipeline import FULL_PIPELINE, run_pipeline
>>> # result = run_pipeline(
>>> #     spec=FULL_PIPELINE,
>>> #     snapshot=...,
>>> #     paths=...,
>>> #     gateway=...,
>>> #     tools=...,
>>> # )
"""

from codeintel.pipeline import run_registry
from codeintel.pipeline.executor import run_pipeline
from codeintel.pipeline.planner import (
    AnalyticsStagePlan,
    GraphsStagePlan,
    IngestionStagePlan,
    PipelinePlan,
    build_pipeline_plan,
)
from codeintel.pipeline.spec import (
    ANALYTICS_ONLY,
    FULL_PIPELINE,
    GRAPHS_ONLY,
    INGEST_ONLY,
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
    "PIPELINE_SPECS",
    "AnalyticsStagePlan",
    "GraphsStagePlan",
    "IngestionStagePlan",
    "PipelinePlan",
    "PipelineSpec",
    "PipelineStage",
    "StageModule",
    "build_pipeline_plan",
    "get_pipeline_spec",
    "list_pipeline_specs",
    "run_pipeline",
    "run_registry",
]
