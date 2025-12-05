"""Unified recipe executor dispatching to domain-specific executors.

This module provides a unified executor that can execute any UnifiedRecipe
by dispatching stages to the appropriate domain executor based on module.

The UnifiedRecipeExecutor coordinates execution across:
- Ingestion plugins (via ingestion runtime)
- Graph plugins (via graph runtime)
- Analytics plugins (via analytics runtime)
- Pipeline steps (via step registry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.recipes.unified import (
    StageModule,
    UnifiedRecipe,
    UnifiedStage,
)

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.execution import RunContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedStageResult:
    """Result of executing a single unified stage.

    Attributes
    ----------
    stage_name
        Name of the executed stage.
    module
        Module that executed the stage.
    success
        Whether all plugins in the stage succeeded.
    plugin_count
        Number of plugins executed.
    success_count
        Number of successful plugins.
    failure_count
        Number of failed plugins.
    skip_count
        Number of skipped plugins.
    duration_ms
        Execution duration in milliseconds.
    errors
        Error messages from failed plugins.
    """

    stage_name: str
    module: StageModule
    success: bool
    plugin_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    skip_count: int = 0
    duration_ms: float = 0.0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnifiedRecipeResult:
    """Result of executing a unified recipe.

    Attributes
    ----------
    recipe_name
        Name of the executed recipe.
    success
        Whether all required stages succeeded.
    stage_results
        Results for each executed stage.
    duration_ms
        Total execution duration in milliseconds.
    error
        Overall error message if execution failed.
    started_at
        When execution started.
    ended_at
        When execution ended.
    """

    recipe_name: str
    success: bool
    stage_results: tuple[UnifiedStageResult, ...] = ()
    duration_ms: float = 0.0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_plugins(self) -> int:
        """Return total plugins across all stages.

        Returns
        -------
        int
            Total plugin count.
        """
        return sum(r.plugin_count for r in self.stage_results)

    @property
    def success_count(self) -> int:
        """Return total successful plugins.

        Returns
        -------
        int
            Success count.
        """
        return sum(r.success_count for r in self.stage_results)

    @property
    def failure_count(self) -> int:
        """Return total failed plugins.

        Returns
        -------
        int
            Failure count.
        """
        return sum(r.failure_count for r in self.stage_results)


@dataclass
class UnifiedExecutorContext:
    """Context for unified recipe execution.

    Provides all dependencies needed for executing recipes across
    all modules.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    run_context
        Optional run context for correlation.
    scratch
        Shared scratch space for inter-plugin communication.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_context: RunContext | None = None
    scratch: PluginScratch = field(default_factory=PluginScratch)


class UnifiedRecipeExecutor:
    """Executor for unified recipes with domain dispatch.

    This executor coordinates execution of unified recipes by dispatching
    each stage to the appropriate domain executor based on the stage's
    module assignment.

    The executor supports:
    - Sequential stage execution
    - Fail-fast or continue-on-error modes
    - Per-stage error isolation
    - Aggregated results across all modules

    Attributes
    ----------
    _log
        Logger instance for this executor.

    Examples
    --------
    >>> ctx = UnifiedExecutorContext(gateway=gw, snapshot=snap)
    >>> executor = UnifiedRecipeExecutor()
    >>> result = executor.execute(recipe, ctx)
    >>> result.success
    True
    """

    def __init__(self) -> None:
        """Initialize the executor with a logger."""
        self._log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(
        self,
        recipe: UnifiedRecipe,
        context: UnifiedExecutorContext,
    ) -> UnifiedRecipeResult:
        """Execute a unified recipe.

        Parameters
        ----------
        recipe
            Recipe to execute.
        context
            Execution context with dependencies.

        Returns
        -------
        UnifiedRecipeResult
            Aggregated execution result.
        """
        started_at = datetime.now(UTC)
        stage_results: list[UnifiedStageResult] = []
        overall_success = True
        overall_error: str | None = None

        try:
            for stage in recipe.stages:
                stage_result = self._execute_stage(stage, recipe, context)
                stage_results.append(stage_result)

                if not stage_result.success and stage.required:
                    overall_success = False
                    if recipe.options.fail_fast:
                        overall_error = f"Required stage '{stage.name}' failed"
                        break

            ended_at = datetime.now(UTC)
            duration_ms = (ended_at - started_at).total_seconds() * 1000

            return UnifiedRecipeResult(
                recipe_name=recipe.name,
                success=overall_success,
                stage_results=tuple(stage_results),
                duration_ms=duration_ms,
                error=overall_error,
                started_at=started_at,
                ended_at=ended_at,
            )

        except (RuntimeError, ValueError, KeyError) as exc:
            ended_at = datetime.now(UTC)
            duration_ms = (ended_at - started_at).total_seconds() * 1000
            return UnifiedRecipeResult(
                recipe_name=recipe.name,
                success=False,
                stage_results=tuple(stage_results),
                duration_ms=duration_ms,
                error=str(exc),
                started_at=started_at,
                ended_at=ended_at,
            )

        finally:
            context.scratch.cleanup()

    def _execute_stage(
        self,
        stage: UnifiedStage,
        recipe: UnifiedRecipe,
        context: UnifiedExecutorContext,
    ) -> UnifiedStageResult:
        """Execute a single stage using the appropriate domain executor.

        Parameters
        ----------
        stage
            Stage to execute.
        recipe
            Parent recipe for configuration.
        context
            Execution context.

        Returns
        -------
        UnifiedStageResult
            Stage execution result.
        """
        self._log.info(
            "unified_executor.stage.start name=%s module=%s plugins=%d recipe=%s",
            stage.name,
            stage.module,
            len(stage.plugins),
            recipe.name,
        )

        started = datetime.now(UTC)
        errors: list[str] = []

        try:
            if stage.module == "ingestion":
                result = _execute_ingestion_stage(stage, context)
            elif stage.module == "graphs":
                result = _execute_graphs_stage(stage, context)
            elif stage.module == "analytics":
                result = _execute_analytics_stage(stage, context)
            elif stage.module == "pipeline":
                result = _execute_pipeline_stage(stage, context)
            else:
                result = _create_skip_result(stage, "Unknown module")

        except (RuntimeError, ValueError, KeyError) as exc:
            ended = datetime.now(UTC)
            duration_ms = (ended - started).total_seconds() * 1000
            errors.append(str(exc))
            result = UnifiedStageResult(
                stage_name=stage.name,
                module=stage.module,
                success=False,
                plugin_count=len(stage.plugins),
                failure_count=len(stage.plugins),
                duration_ms=duration_ms,
                errors=tuple(errors),
            )

        self._log.info(
            "unified_executor.stage.end name=%s success=%s duration_ms=%.2f",
            stage.name,
            result.success,
            result.duration_ms,
        )

        return result


def _execute_ingestion_stage(
    stage: UnifiedStage,
    context: UnifiedExecutorContext,
) -> UnifiedStageResult:
    """Execute an ingestion stage.

    Parameters
    ----------
    stage
        Ingestion stage to execute.
    context
        Execution context.

    Returns
    -------
    UnifiedStageResult
        Stage result.
    """
    # Placeholder - actual implementation would delegate to ingestion runtime
    log.debug("ingestion_stage context.snapshot=%s", context.snapshot)
    return _create_placeholder_result(stage, "ingestion")


def _execute_graphs_stage(
    stage: UnifiedStage,
    context: UnifiedExecutorContext,
) -> UnifiedStageResult:
    """Execute a graphs stage.

    Parameters
    ----------
    stage
        Graphs stage to execute.
    context
        Execution context.

    Returns
    -------
    UnifiedStageResult
        Stage result.
    """
    # Placeholder - actual implementation would delegate to graph runtime
    log.debug("graphs_stage context.snapshot=%s", context.snapshot)
    return _create_placeholder_result(stage, "graphs")


def _execute_analytics_stage(
    stage: UnifiedStage,
    context: UnifiedExecutorContext,
) -> UnifiedStageResult:
    """Execute an analytics stage.

    Parameters
    ----------
    stage
        Analytics stage to execute.
    context
        Execution context.

    Returns
    -------
    UnifiedStageResult
        Stage result.
    """
    # Placeholder - actual implementation would delegate to analytics runtime
    log.debug("analytics_stage context.snapshot=%s", context.snapshot)
    return _create_placeholder_result(stage, "analytics")


def _execute_pipeline_stage(
    stage: UnifiedStage,
    context: UnifiedExecutorContext,
) -> UnifiedStageResult:
    """Execute a pipeline step stage.

    Parameters
    ----------
    stage
        Pipeline stage to execute.
    context
        Execution context.

    Returns
    -------
    UnifiedStageResult
        Stage result.
    """
    # Placeholder - actual implementation would delegate to step registry
    log.debug("pipeline_stage context.snapshot=%s", context.snapshot)
    return _create_placeholder_result(stage, "pipeline")


def _create_placeholder_result(
    stage: UnifiedStage,
    module: StageModule,
) -> UnifiedStageResult:
    """Create a placeholder result for unimplemented stages.

    This is used during development to return success for stages
    that don't yet have full executor integration.

    Parameters
    ----------
    stage
        Stage being executed.
    module
        Module type.

    Returns
    -------
    UnifiedStageResult
        Placeholder success result.
    """
    return UnifiedStageResult(
        stage_name=stage.name,
        module=module,
        success=True,
        plugin_count=len(stage.plugins),
        success_count=len(stage.plugins),
        duration_ms=0.0,
    )


def _create_skip_result(
    stage: UnifiedStage,
    reason: str,
) -> UnifiedStageResult:
    """Create a skip result for a stage.

    Parameters
    ----------
    stage
        Stage being skipped.
    reason
        Skip reason.

    Returns
    -------
    UnifiedStageResult
        Skip result.
    """
    return UnifiedStageResult(
        stage_name=stage.name,
        module=stage.module,
        success=True,  # Skipped stages don't fail
        plugin_count=len(stage.plugins),
        skip_count=len(stage.plugins),
        duration_ms=0.0,
        errors=(reason,),
    )


__all__ = [
    "UnifiedExecutorContext",
    "UnifiedRecipeExecutor",
    "UnifiedRecipeResult",
    "UnifiedStageResult",
]
