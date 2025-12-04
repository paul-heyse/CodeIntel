"""Recipe execution engine with parallel stage support.

This module provides the executor for running ingestion recipes,
with support for parallel plugin execution, failure handling,
and observability.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.plugins.protocol import (
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestRuntimeScratch,
)
from codeintel.ingestion.plugins.registry import (
    IngestPluginRegistry,
    PlanOptions,
    get_ingest_registry,
)
from codeintel.ingestion.recipes.dsl import (
    IngestRecipe,
    RecipeExecutionResult,
    RecipeOptions,
    RecipeStage,
    RecipeStageResult,
)
from codeintel.ingestion.resources.modules import ModuleProvider
from codeintel.ingestion.resources.registry import ResourceRegistry
from codeintel.ingestion.resources.tools import ToolsProvider
from codeintel.ingestion.resources.tracker import TrackerConfig, TrackerProvider
from codeintel.ingestion.tracker import ChangeTracker
from codeintel.storage.tracking import PipelineStatus, PipelineStepRecord, StepStatus

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.ingestion.core.runs import IngestRunSink
    from codeintel.ingestion.engine.infrastructure import ToolRunner
    from codeintel.ingestion.engine.service import ToolService
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.runtime import RunContext
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.tracking import PipelineRunTracking

log = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """Configuration for recipe execution.

    Attributes
    ----------
    registry
        Plugin registry to use.
    scratch
        Shared scratch space.
    run_id
        Unique run identifier.
    enable_parallel
        Whether to enable parallel execution.
    max_workers
        Maximum thread workers for parallel stages.
    timeout_s
        Default timeout per plugin in seconds.
    """

    registry: IngestPluginRegistry = field(default_factory=get_ingest_registry)
    scratch: IngestRuntimeScratch = field(default_factory=IngestRuntimeScratch)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    enable_parallel: bool = True
    max_workers: int = 4
    timeout_s: int | None = None


@dataclass
class RecipeExecutorContext:
    """Execution context for recipe execution.

    Encapsulates all dependencies and services needed by the RecipeExecutor,
    reducing parameter count in initialization.

    Attributes
    ----------
    gateway
        StorageGateway for database access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths configuration.
    tools
        Tools configuration.
    code_profile
        Code scanning profile.
    config_profile
        Config scanning profile.
    tool_runner
        Optional shared tool runner.
    tool_service
        Optional shared tool service.
    change_tracker
        Optional change tracker for incremental ingestion.
    ingest_run_sink
        Optional sink for recording run metrics.
    run_context
        Optional unified run context for cross-engine correlation.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    tools: ToolsConfig
    code_profile: ScanProfile
    config_profile: ScanProfile
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    change_tracker: ChangeTracker | None = None
    ingest_run_sink: IngestRunSink | None = None
    run_context: RunContext | None = None


@dataclass
class PluginExecutionRecord:
    """Record of a single plugin execution.

    Attributes
    ----------
    plugin_name
        Name of the executed plugin.
    result
        Execution result.
    duration_s
        Execution duration in seconds.
    error
        Exception if execution failed.
    """

    plugin_name: str
    result: IngestPluginResult | None = None
    duration_s: float = 0.0
    error: Exception | None = None


class RecipeExecutor:
    """Execute ingestion recipes with parallel stage support.

    The executor runs plugins in the order defined by recipe stages,
    with optional parallelism within stages.
    """

    def __init__(
        self,
        context: RecipeExecutorContext,
        config: ExecutorConfig | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        context
            Execution context with dependencies and services.
        config
            Executor configuration.
        """
        self._gateway = context.gateway
        self._snapshot = context.snapshot
        self._paths = context.paths
        self._tools = context.tools
        self._code_profile = context.code_profile
        self._config_profile = context.config_profile
        self._tool_runner = context.tool_runner
        self._tool_service = context.tool_service
        self._change_tracker = context.change_tracker
        self._ingest_run_sink = context.ingest_run_sink
        self._run_context = context.run_context
        self._config = config or ExecutorConfig()

    def execute(self, recipe: IngestRecipe) -> RecipeExecutionResult:
        """Execute a recipe.

        Parameters
        ----------
        recipe
            Recipe to execute.

        Returns
        -------
        RecipeExecutionResult
            Execution result with stage details.
        """
        start_time = time.perf_counter()
        stage_results: list[RecipeStageResult] = []
        skipped_stages: list[str] = []
        error: str | None = None
        success = True

        log.info(
            "Starting recipe execution: recipe=%s run_id=%s stages=%d",
            recipe.name,
            self._config.run_id,
            len(recipe.stages),
        )

        # Build execution plan
        try:
            plan = self._build_plan(recipe)
        except ValueError:
            log.exception("Failed to build execution plan")
            return RecipeExecutionResult(
                recipe=recipe,
                success=False,
                duration_s=time.perf_counter() - start_time,
                error="Failed to build execution plan",
            )

        # Execute stages
        for stage in recipe.stages:
            # Get plugins for this stage from the plan
            stage_plugins = [
                plugin for plugin in plan.plugins if plugin.metadata.name in stage.plugins
            ]

            if not stage_plugins:
                log.info("Skipping empty stage: %s", stage.name)
                skipped_stages.append(stage.name)
                continue

            stage_result = self._execute_stage(stage, stage_plugins, recipe.options)
            stage_results.append(stage_result)

            if not stage_result.success:
                if stage.required and recipe.options.fail_fast:
                    success = False
                    error = f"Required stage '{stage.name}' failed"
                    log.error(error)
                    break
                if stage.required:
                    success = False
                    log.warning("Required stage '%s' failed but continuing", stage.name)

        # Cleanup
        self._config.scratch.cleanup()

        duration_s = time.perf_counter() - start_time
        log.info(
            "Recipe execution completed: recipe=%s success=%s duration=%.2fs",
            recipe.name,
            success,
            duration_s,
        )

        return RecipeExecutionResult(
            recipe=recipe,
            success=success,
            stage_results=tuple(stage_results),
            skipped_stages=tuple(skipped_stages),
            duration_s=duration_s,
            error=error,
        )

    def _build_plan(self, recipe: IngestRecipe) -> IngestPluginPlan:
        """Build execution plan from recipe.

        Parameters
        ----------
        recipe
            Recipe to plan.

        Returns
        -------
        IngestPluginPlan
            Resolved execution plan.
        """
        # Collect all plugins from stages
        plugin_names = list(recipe.all_plugins)

        # Apply enabled/disabled overrides
        disabled = set(recipe.disabled_plugins)

        return self._config.registry.plan(
            PlanOptions(
                plugin_names=plugin_names,
                disabled=tuple(disabled),
            )
        )

    def _execute_stage(
        self,
        stage: RecipeStage,
        plugins: Sequence[IngestPluginProtocol],
        _options: RecipeOptions,
    ) -> RecipeStageResult:
        """Execute a single stage.

        Parameters
        ----------
        stage
            Stage to execute.
        plugins
            Plugins to run in this stage.
        _options
            Recipe options placeholder (reserved for future use).

        Returns
        -------
        RecipeStageResult
            Stage execution result.
        """
        start_time = time.perf_counter()
        plugin_results: dict[str, object] = {}
        success = True

        log.info(
            "Executing stage: name=%s plugins=%s parallel=%s",
            stage.name,
            [p.metadata.name for p in plugins],
            stage.parallel,
        )

        if stage.parallel and self._config.enable_parallel and len(plugins) > 1:
            records = self._execute_parallel(plugins, stage.timeout_s)
        else:
            records = self._execute_sequential(plugins, stage.timeout_s)

        for record in records:
            plugin_results[record.plugin_name] = {
                "success": record.result.success if record.result else False,
                "skipped": record.result.skipped if record.result else False,
                "error": record.result.error if record.result else str(record.error),
                "duration_s": record.duration_s,
            }

            # Mark as failure unless explicitly skipped
            is_skip = record.result is not None and record.result.skipped
            if (record.result is None or not record.result.success) and not is_skip:
                success = False

        duration_s = time.perf_counter() - start_time
        log.info(
            "Stage completed: name=%s success=%s duration=%.2fs",
            stage.name,
            success,
            duration_s,
        )

        return RecipeStageResult(
            stage=stage,
            success=success,
            plugin_results=plugin_results,
            duration_s=duration_s,
        )

    def _execute_sequential(
        self,
        plugins: Sequence[IngestPluginProtocol],
        timeout_s: int | None,
    ) -> list[PluginExecutionRecord]:
        """Execute plugins sequentially.

        Parameters
        ----------
        plugins
            Plugins to execute.
        timeout_s
            Timeout per plugin.

        Returns
        -------
        list[PluginExecutionRecord]
            Execution records.
        """
        records: list[PluginExecutionRecord] = []

        for plugin in plugins:
            record = self._execute_single_plugin(plugin, timeout_s)
            records.append(record)

            # Update change tracker if repo_scan completed
            if plugin.metadata.name == "repo_scan" and record.result and record.result.success:
                tracker = self._config.scratch.consume("change_tracker")
                if tracker is not None and isinstance(tracker, ChangeTracker):
                    self._change_tracker = tracker

        return records

    def _execute_parallel(
        self,
        plugins: Sequence[IngestPluginProtocol],
        timeout_s: int | None,
    ) -> list[PluginExecutionRecord]:
        """Execute plugins in parallel using threads.

        Parameters
        ----------
        plugins
            Plugins to execute.
        timeout_s
            Timeout per plugin.

        Returns
        -------
        list[PluginExecutionRecord]
            Execution records in completion order.
        """
        records: list[PluginExecutionRecord] = []
        max_workers = min(len(plugins), self._config.max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._execute_single_plugin, plugin, timeout_s): plugin
                for plugin in plugins
            }

            for future in as_completed(futures):
                plugin = futures[future]
                try:
                    record = future.result()
                    records.append(record)
                except Exception as exc:
                    log.exception("Plugin execution failed: %s", plugin.metadata.name)
                    records.append(
                        PluginExecutionRecord(
                            plugin_name=plugin.metadata.name,
                            error=exc,
                        )
                    )

        return records

    def _execute_single_plugin(
        self,
        plugin: IngestPluginProtocol,
        _timeout_s: int | None,
    ) -> PluginExecutionRecord:
        """Execute a single plugin.

        Parameters
        ----------
        plugin
            Plugin to execute.
        _timeout_s
            Timeout in seconds (reserved for future use).

        Returns
        -------
        PluginExecutionRecord
            Execution record.
        """
        name = plugin.metadata.name
        start_time = time.perf_counter()

        log.info("Executing plugin: %s", name)

        try:
            ctx = self._build_context(plugin)
            result = plugin.execute(ctx)
            duration_s = time.perf_counter() - start_time

            log.info(
                "Plugin completed: name=%s success=%s skipped=%s duration=%.2fs",
                name,
                result.success,
                result.skipped,
                duration_s,
            )

            return PluginExecutionRecord(
                plugin_name=name,
                result=result,
                duration_s=duration_s,
            )

        except Exception as exc:
            duration_s = time.perf_counter() - start_time
            log.exception("Plugin failed: name=%s duration=%.2fs", name, duration_s)

            return PluginExecutionRecord(
                plugin_name=name,
                result=IngestPluginResult.fail(str(exc), error_kind=type(exc).__name__),
                duration_s=duration_s,
                error=exc,
            )

    def _build_context(self, plugin: IngestPluginProtocol) -> IngestExecutionContext:
        """Build execution context for a plugin.

        Parameters
        ----------
        plugin
            Plugin to build context for.

        Returns
        -------
        IngestExecutionContext
            Plugin execution context.
        """
        # Build resource registry with available providers
        resources = self._build_resource_registry()

        return IngestExecutionContext(
            gateway=self._gateway,
            snapshot=self._snapshot,
            paths=self._paths,
            tools=self._tools,
            code_profile=self._code_profile,
            config_profile=self._config_profile,
            resources=resources,
            scratch=self._config.scratch,
            plugin_name=plugin.metadata.name,
            run_id=self._config.run_id,
            run_context=self._run_context,
        )

    def _build_resource_registry(self) -> ResourceRegistry:
        """Build resource registry with available providers.

        Returns
        -------
        ResourceRegistry
            Registry with tracker and tools providers.
        """
        registry = ResourceRegistry()

        # Register tracker provider if we have tracker state
        tracker_config = TrackerConfig(
            scratch=self._config.scratch,
            profile=self._code_profile,
            full_rebuild=False,
        )
        tracker_provider = TrackerProvider(
            gateway=self._gateway,
            snapshot=self._snapshot,
            config=tracker_config,
        )
        registry.register(TrackerProvider, tracker_provider)

        # Register tools provider with config and cache dir
        tools_provider = ToolsProvider(
            tools_config=self._tools,
            cache_dir=self._paths.tool_cache,
            runner=self._tool_runner,
            service=self._tool_service,
        )
        registry.register(ToolsProvider, tools_provider)

        # Register module provider for module access
        module_provider = ModuleProvider(
            gateway=self._gateway,
            snapshot=self._snapshot,
            profile=self._code_profile,
        )
        registry.register(ModuleProvider, module_provider)

        return registry


def execute_recipe(
    recipe: IngestRecipe,
    context: RecipeExecutorContext,
    config: ExecutorConfig | None = None,
) -> RecipeExecutionResult:
    """Execute a recipe with the given context.

    Parameters
    ----------
    recipe
        Recipe to execute.
    context
        Execution context with dependencies and services.
    config
        Optional executor configuration.

    Returns
    -------
    RecipeExecutionResult
        Execution result.
    """
    executor = RecipeExecutor(context, config)
    return executor.execute(recipe)


def execute_recipe_for_context(
    recipe: IngestRecipe,
    run_context: RunContext,
    context: RecipeExecutorContext,
    config: ExecutorConfig | None = None,
) -> RecipeExecutionResult:
    """Execute a recipe with unified RunContext.

    This is the preferred entrypoint that accepts a unified RunContext
    for consistent run identity across all engines. It records run and
    step metadata to the pipeline registry.

    Parameters
    ----------
    recipe
        Recipe to execute.
    run_context
        Unified run context for cross-engine correlation.
    context
        Execution context with dependencies and services.
    config
        Optional executor configuration.

    Returns
    -------
    RecipeExecutionResult
        Execution result.

    Examples
    --------
    >>> from codeintel.runtime import new_run_context
    >>> from codeintel.config.primitives import SnapshotRef
    >>> from pathlib import Path
    >>> # Create unified context
    >>> snapshot = SnapshotRef(repo="org/repo", commit="abc123", repo_root=Path("/tmp"))
    >>> run_ctx = new_run_context(snapshot=snapshot, kind="ingest", trigger="cli")
    >>> # Build executor context with run_context
    >>> # context = RecipeExecutorContext(..., run_context=run_ctx)
    >>> # result = execute_recipe_for_context(recipe, run_ctx, context)
    """
    runs = context.gateway.runs

    # Start the run in the registry
    runs.start_run(
        run_context,
        pipeline_name=f"ingest:{recipe.name}",
    )

    # Ensure context has the run_context set
    context_with_run = replace(context, run_context=run_context)

    # Execute the recipe
    result = execute_recipe(recipe, context_with_run, config)

    # Record steps from stage results
    _record_ingestion_steps(runs, run_context.run_id, result)

    # Determine overall status
    status: PipelineStatus
    error_summary: str | None = None
    if result.success:
        status = "succeeded"
    elif result.error:
        status = "failed"
        error_summary = result.error
    else:
        # Some stages failed but not all
        status = "partial"

    # Complete the run
    runs.complete_run(
        run_context.run_id,
        status=status,
        error_summary=error_summary,
    )

    return result


def _record_ingestion_steps(
    runs: PipelineRunTracking,
    run_id: str,
    result: RecipeExecutionResult,
) -> None:
    """Record step records from ingestion results.

    Parameters
    ----------
    runs
        Pipeline run tracking accessor from gateway.
    run_id
        Run identifier.
    result
        Recipe execution result.
    """
    for stage_result in result.stage_results:
        stage_name = stage_result.stage.name
        for plugin_name, plugin_data in stage_result.plugin_results.items():
            # Extract status from plugin result
            if isinstance(plugin_data, dict):
                success = plugin_data.get("success", False)
                skipped = plugin_data.get("skipped", False)
                error = plugin_data.get("error")
            else:
                success = False
                skipped = False
                error = None

            step_status: StepStatus
            if skipped:
                step_status = "skipped"
            elif success:
                step_status = "succeeded"
            else:
                step_status = "failed"

            # Use current time as approximation since we don't have exact timestamps
            now = datetime.now(tz=UTC)
            extra: dict[str, object] | None = None
            if error:
                extra = {"error": error}

            runs.record_step(
                PipelineStepRecord(
                    run_id=run_id,
                    module="ingestion",
                    stage=stage_name,
                    name=plugin_name,
                    status=step_status,
                    started_at=now,
                    completed_at=now,
                    row_counts=None,  # Not available in current result structure
                    extra=extra,
                ),
            )


__all__ = [
    "ExecutorConfig",
    "PluginExecutionRecord",
    "RecipeExecutor",
    "RecipeExecutorContext",
    "execute_recipe",
    "execute_recipe_for_context",
]
