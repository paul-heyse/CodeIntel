"""Pipeline orchestration and export utilities for batch CodeIntel operations.

This package provides:

- **Declarative specs**: Define pipelines as ordered stages via :class:`PipelineSpec`
- **Planning**: Build execution plans via :func:`build_pipeline_plan`
- **Execution**: Run pipelines via :func:`run_pipeline`
- **CLI adapter**: Bridge CLI arguments to plan options via :class:`CliPipelineArgs`
- **Gateway caching**: Manage gateway lifecycle via :mod:`codeintel.storage.gateway_cache`
- **Config resolution**: Environment-aware config helpers via :mod:`config_resolver`
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

CLI Usage
---------
>>> from codeintel.pipeline import CliPipelineArgs, get_gateway, close_gateways
>>> from codeintel.storage.gateway import StorageConfig
>>> # cli_args = CliPipelineArgs(repo_root=..., repo=..., commit=..., ...)
>>> # gateway = get_gateway(StorageConfig.for_ingest(cli_args.db_path))
>>> # try:
>>> #     options = cli_args.to_plan_options(gateway, tools)
>>> #     run_pipeline(spec=FULL_PIPELINE, options=options)
>>> # finally:
>>> #     close_gateways()

Operation-Driven Orchestration
------------------------------
>>> from codeintel.pipeline import build_pipeline_for_operation
>>> # spec = build_pipeline_for_operation("function.summary", snapshot)
>>> # assert spec.id == "full"  # Requires graphs, so needs full pipeline
"""

# Primary API (spec-based execution)
# CLI adapter and infrastructure
from codeintel.pipeline.cli_adapter import CliPipelineArgs
from codeintel.pipeline.config_resolver import (
    resolve_graph_backend,
    resolve_scan_profiles,
    resolve_tools_config,
)
from codeintel.pipeline.execution import tracking as run_registry
from codeintel.pipeline.execution.runner import run_pipeline
from codeintel.pipeline.execution.step_runner import (
    DEFAULT_BUILD_SUBDIR,
    ExportHooks,
    HistoryTimeseriesParams,
    run_export_docs,
    run_history_timeseries,
)
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

# Gateway caching (from storage layer)
from codeintel.storage.gateway_cache import (
    GatewayCache,
    close_gateways,
    gateway_cache_stats,
    get_gateway,
)

__all__ = [
    "ANALYTICS_ONLY",
    "DEFAULT_BUILD_SUBDIR",
    "FULL_PIPELINE",
    "GRAPHS_ONLY",
    "INGEST_ONLY",
    "NOOP_PIPELINE",
    "PIPELINE_SPECS",
    "AnalyticsStagePlan",
    "CliPipelineArgs",
    "ExportHooks",
    "GatewayCache",
    "GraphsStagePlan",
    "HistoryTimeseriesParams",
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
    "close_gateways",
    "ensure_prerequisites_for_operation",
    "gateway_cache_stats",
    "get_gateway",
    "get_pipeline_spec",
    "list_pipeline_specs",
    "resolve_graph_backend",
    "resolve_scan_profiles",
    "resolve_tools_config",
    "run_export_docs",
    "run_history_timeseries",
    "run_pipeline",
    "run_registry",
]
