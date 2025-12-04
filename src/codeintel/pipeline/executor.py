"""Unified pipeline executor.

This module provides the main entrypoint for executing unified pipelines
across ingestion, graphs, and analytics stages. The executor manages run
tracking, stage dispatch, and failure handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.core.pipeline_bridge import run_analytics_plugins
from codeintel.graphs.runtime.executor import run_graph_plugins
from codeintel.ingestion.recipes.executor import execute_recipe
from codeintel.pipeline.planner import (
    AnalyticsStagePlan,
    GraphsStagePlan,
    IngestionStagePlan,
    PipelinePlan,
    PipelinePlanOptions,
    build_pipeline_plan,
)
from codeintel.pipeline.spec import PipelineSpec, PipelineStage
from codeintel.storage.tracking import PipelineStatus, PipelineStepRecord

if TYPE_CHECKING:
    from codeintel.storage.tracking import (
        PipelineRunRecord,
        PipelineRunTracking,
        StepStatus,
    )

log = logging.getLogger(__name__)


def _now() -> datetime:
    """Return current UTC timestamp.

    Returns
    -------
    datetime
        Current datetime with UTC timezone.
    """
    return datetime.now(tz=UTC)


@dataclass(frozen=True)
class StageStepContext:
    """Tracking context for a single stage step."""

    runs: PipelineRunTracking
    run_id: str
    stage: PipelineStage
    started_at: datetime


# -----------------------------------------------------------------------------
# Stage Step Recording
# -----------------------------------------------------------------------------


def _start_stage_step(
    runs: PipelineRunTracking,
    run_id: str,
    stage: PipelineStage,
) -> datetime:
    """Record start of a stage-level step.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor.
    run_id
        Run identifier.
    stage
        Pipeline stage being started.

    Returns
    -------
    datetime
        Start timestamp for use in completion.
    """
    started_at = _now()
    runs.record_step(
        PipelineStepRecord(
            run_id=run_id,
            module=stage.module,
            stage="orchestrator",
            name=stage.name,
            status="running",
            started_at=started_at,
            completed_at=None,
            row_counts=None,
            extra=None,
        )
    )
    return started_at


def _complete_stage_step(
    ctx: StageStepContext,
    status: StepStatus,
    *,
    error: str | None = None,
) -> None:
    """Record completion of a stage-level step.

    Parameters
    ----------
    ctx
        Stage context including run metadata.
    status
        Final status of the step.
    error
        Optional error message if failed.
    """
    extra: dict[str, object] | None = None
    if error:
        extra = {"error": error}

    ctx.runs.record_step(
        PipelineStepRecord(
            run_id=ctx.run_id,
            module=ctx.stage.module,
            stage="orchestrator",
            name=ctx.stage.name,
            status=status,
            started_at=ctx.started_at,
            completed_at=_now(),
            row_counts=None,
            extra=extra,
        )
    )


# -----------------------------------------------------------------------------
# Stage Execution
# -----------------------------------------------------------------------------


def _execute_ingestion_stage(
    plan: IngestionStagePlan,
) -> None:
    """Execute an ingestion stage.

    Parameters
    ----------
    plan
        Ingestion stage plan with recipe and context.

    Raises
    ------
    RuntimeError
        If recipe execution fails.
    """
    result = execute_recipe(
        recipe=plan.recipe,
        context=plan.context,
        config=None,
    )

    if not result.success:
        error_msg = result.error or "Ingestion failed"
        raise RuntimeError(error_msg)


def _execute_graphs_stage(
    plan: GraphsStagePlan,
) -> None:
    """Execute a graphs stage.

    Parameters
    ----------
    plan
        Graphs stage plan with plugin plan and context.

    Raises
    ------
    RuntimeError
        If any graph plugin fails.
    """
    report = run_graph_plugins(
        plan=plan.plan,
        context=plan.context,
    )

    if report.fatal_error or report.failure_count > 0:
        failed_plugins = [r.plugin_name for r in report.records if r.status == "failed"]
        error_msg = f"Graph plugins failed: {', '.join(failed_plugins)}"
        raise RuntimeError(error_msg)


def _execute_analytics_stage(
    plan: AnalyticsStagePlan,
) -> None:
    """Execute an analytics stage.

    Parameters
    ----------
    plan
        Analytics stage plan with plugin plan and context.

    Raises
    ------
    RuntimeError
        If any analytics plugin fails.
    """
    report = run_analytics_plugins(
        plan=plan.plan,
        run_context=plan.context,
        enable_middleware=True,
    )

    failed_records = [r for r in report.records if r.status == "failed"]
    if failed_records:
        failed_names = [r.name for r in failed_records]
        error_msg = f"Analytics plugins failed: {', '.join(failed_names)}"
        raise RuntimeError(error_msg)


# -----------------------------------------------------------------------------
# Main Executor Function
# -----------------------------------------------------------------------------


def run_pipeline(
    *,
    spec: PipelineSpec,
    options: PipelinePlanOptions,
) -> PipelineRunRecord:
    """Execute a unified pipeline over ingestion, graphs, and/or analytics.

    This is the main entrypoint for executing unified pipelines. It creates
    a single RunContext shared across all stages, manages run tracking, and
    handles fail-fast behavior for required stages.

    Parameters
    ----------
    spec
        Declarative pipeline specification (e.g., FULL_PIPELINE).
    options
        Bundled plan/execution options (snapshot, paths, gateway, tools, trigger).

    Returns
    -------
    PipelineRunRecord
        Final run record from the run tracking table.

    Raises
    ------
    RuntimeError
        If the run record cannot be fetched after execution.

    Examples
    --------
    >>> from codeintel.pipeline.spec import FULL_PIPELINE
    >>> from codeintel.config.primitives import SnapshotRef, BuildPaths
    >>> # result = run_pipeline(
    >>> #     spec=FULL_PIPELINE,
    >>> #     snapshot=snapshot,
    >>> #     paths=paths,
    >>> #     gateway=gateway,
    >>> #     tools=tools,
    >>> # )
    >>> # assert result.status in ("succeeded", "failed")
    """
    runs = options.gateway.runs

    # Build the execution plan
    plan = build_pipeline_plan(
        spec=spec,
        options=options,
    )
    run_ctx = plan.run_context
    run_id = run_ctx.run_id

    log.info(
        "pipeline.executor.start spec=%s run_id=%s stages=%d",
        spec.id,
        run_id,
        len(spec.stages),
    )

    # Start the run
    runs.start_run(
        run_ctx,
        pipeline_name=spec.id,
        status="running",
    )

    overall_status: PipelineStatus = "succeeded"
    last_error: str | None = None

    # Execute stages in order
    for stage in spec.stages:
        stage_plan = _get_stage_plan(plan, stage)
        if stage_plan is None:
            log.warning(
                "pipeline.executor.skip_stage stage=%s reason=no_plan",
                stage.name,
            )
            continue

        started_at = _start_stage_step(runs, run_id, stage)
        step_ctx = StageStepContext(runs=runs, run_id=run_id, stage=stage, started_at=started_at)

        log.info(
            "pipeline.executor.stage.start module=%s name=%s required=%s",
            stage.module,
            stage.name,
            stage.required,
        )

        try:
            _execute_stage(stage_plan)
            _complete_stage_step(step_ctx, "succeeded")
            log.info(
                "pipeline.executor.stage.complete module=%s name=%s status=succeeded",
                stage.module,
                stage.name,
            )
        except Exception as exc:
            error_msg = f"Stage {stage.module}:{stage.name} failed: {exc}"
            last_error = error_msg
            _complete_stage_step(step_ctx, "failed", error=str(exc))

            log.exception(
                "pipeline.executor.stage.failed module=%s name=%s",
                stage.module,
                stage.name,
            )

            if stage.required:
                overall_status = "failed"
                # Fail-fast: stop executing further stages
                runs.complete_run(run_id, status=overall_status, error_summary=error_msg)
                run = runs.fetch_run(run_id)
                if run is None:
                    message = f"Failed to fetch run record for run_id={run_id}"
                    raise RuntimeError(message) from exc
                return run

    # All stages completed (or non-required failures occurred)
    runs.complete_run(run_id, status=overall_status, error_summary=last_error)

    log.info(
        "pipeline.executor.complete spec=%s run_id=%s status=%s",
        spec.id,
        run_id,
        overall_status,
    )

    run = runs.fetch_run(run_id)
    if run is None:
        message = f"Failed to fetch run record for run_id={run_id}"
        raise RuntimeError(message)
    return run


def _get_stage_plan(
    plan: PipelinePlan,
    stage: PipelineStage,
) -> IngestionStagePlan | GraphsStagePlan | AnalyticsStagePlan | None:
    """Get the appropriate stage plan from the pipeline plan.

    Parameters
    ----------
    plan
        Pipeline plan with all stage plans.
    stage
        Stage to get plan for.

    Returns
    -------
    IngestionStagePlan | GraphsStagePlan | AnalyticsStagePlan | None
        Stage plan if available, None otherwise.
    """
    if stage.module == "ingestion":
        return plan.ingestion
    if stage.module == "graphs":
        return plan.graphs
    if stage.module == "analytics":
        return plan.analytics
    return None


def _execute_stage(
    stage_plan: IngestionStagePlan | GraphsStagePlan | AnalyticsStagePlan,
) -> None:
    """Execute a stage plan by dispatching to the appropriate engine.

    Parameters
    ----------
    stage_plan
        Stage plan to execute.

    Raises
    ------
    TypeError
        If stage_plan is an unknown type.
    """
    if isinstance(stage_plan, IngestionStagePlan):
        _execute_ingestion_stage(stage_plan)
    elif isinstance(stage_plan, GraphsStagePlan):
        _execute_graphs_stage(stage_plan)
    elif isinstance(stage_plan, AnalyticsStagePlan):
        _execute_analytics_stage(stage_plan)
    else:
        message = f"Unknown stage plan type: {type(stage_plan)}"
        raise TypeError(message)


__all__ = [
    "run_pipeline",
]
