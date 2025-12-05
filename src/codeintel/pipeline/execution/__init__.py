"""Pipeline execution infrastructure.

This package contains execution utilities:

Primary API (Spec-Based)
------------------------
- :func:`run_pipeline`: Execute a spec-based pipeline plan

Other Utilities
---------------
- :class:`PipelineContext`: Execution context for pipeline steps
- :func:`run_history_timeseries`: Execute history timeseries analytics
- :func:`run_export_docs`: Export Parquet/JSONL artifacts
- Run tracking types from codeintel.storage.tracking (re-exported)

Gateway Caching
---------------
Gateway caching functions are available from :mod:`codeintel.storage.gateway_cache`.
"""

from __future__ import annotations

from codeintel.pipeline.execution.context import PipelineContext
from codeintel.pipeline.execution.runner import run_pipeline
from codeintel.pipeline.execution.step_runner import (
    ExportHooks,
    HistoryTimeseriesParams,
    run_export_docs,
    run_history_timeseries,
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
from codeintel.storage.gateway_cache import (
    close_gateways,
    gateway_cache_stats,
    get_gateway,
)

__all__ = [
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
    "close_gateways",
    "gateway_cache_stats",
    "get_gateway",
    "run_export_docs",
    "run_history_timeseries",
    "run_pipeline",
]
