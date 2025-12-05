"""Pipeline execution infrastructure.

This package contains execution utilities:

- run_pipeline: Execute a spec-based pipeline plan
- run_full_pipeline: Execute step-based pipeline (legacy entry point)
- PipelineContext: Execution context for pipeline steps
- ExportArgs: Configuration for step-based pipeline execution
- Run tracking types from codeintel.storage.tracking (re-exported)
"""

from __future__ import annotations

from codeintel.pipeline.execution.context import PipelineContext
from codeintel.pipeline.execution.runner import run_pipeline
from codeintel.pipeline.execution.step_runner import (
    ExportArgs,
    ExportHooks,
    HistoryTimeseriesParams,
    build_pipeline_context,
    close_gateways,
    gateway_cache_stats,
    run_export_docs,
    run_full_pipeline,
    run_history_timeseries,
    run_pipeline_with_retries,
)
from codeintel.pipeline.execution.tracking import (
    ModuleKind,
    PipelineRunRecord,
    PipelineRunTracking,
    PipelineStatus,
    PipelineStepRecord,
    StepStatus,
)
from codeintel.pipeline.steps.base import PipelineStep, StepPhase

__all__ = [
    "ExportArgs",
    "ExportHooks",
    "HistoryTimeseriesParams",
    "ModuleKind",
    "PipelineContext",
    "PipelineRunRecord",
    "PipelineRunTracking",
    "PipelineStatus",
    "PipelineStep",
    "PipelineStepRecord",
    "StepPhase",
    "StepStatus",
    "build_pipeline_context",
    "close_gateways",
    "gateway_cache_stats",
    "run_export_docs",
    "run_full_pipeline",
    "run_history_timeseries",
    "run_pipeline",
    "run_pipeline_with_retries",
]
