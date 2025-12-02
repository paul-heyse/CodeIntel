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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from codeintel.ingestion.change_tracker import ChangeTracker
from codeintel.ingestion.plugins.protocol import (
    IngestPluginContext,
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestRuntimeScratch,
)
from codeintel.ingestion.plugins.registry import (
    IngestPluginRegistry,
    get_ingest_registry,
)
from codeintel.ingestion.recipes.dsl import (
    IngestRecipe,
    RecipeExecutionResult,
    RecipeOptions,
    RecipeStage,
    RecipeStageResult,
)

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.ingestion.ingest_runs import IngestRunSink
    from codeintel.ingestion.source_scanner import ScanProfile
    from codeintel.ingestion.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway

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

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        paths: BuildPaths,
        tools: ToolsConfig,
        code_profile: ScanProfile,
        config_profile: ScanProfile,
        *,
        tool_runner: ToolRunner | None = None,
        tool_service: ToolService | None = None,
        change_tracker: ChangeTracker | None = None,
        ingest_run_sink: IngestRunSink | None = None,
        config: ExecutorConfig | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
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
        config
            Executor configuration.
        """
        self._gateway = gateway
        self._snapshot = snapshot
        self._paths = paths
        self._tools = tools
        self._code_profile = code_profile
        self._config_profile = config_profile
        self._tool_runner = tool_runner
        self._tool_service = tool_service
        self._change_tracker = change_tracker
        self._ingest_run_sink = ingest_run_sink
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
            plugin_names=plugin_names,
            disabled=tuple(disabled),
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

    def _build_context(self, plugin: IngestPluginProtocol) -> IngestPluginContext:
        """Build execution context for a plugin.

        Parameters
        ----------
        plugin
            Plugin to build context for.

        Returns
        -------
        IngestPluginContext
            Plugin execution context.
        """
        return IngestPluginContext(
            gateway=self._gateway,
            snapshot=self._snapshot,
            paths=self._paths,
            tools=self._tools,
            code_profile=self._code_profile,
            config_profile=self._config_profile,
            tool_runner=self._tool_runner,
            tool_service=self._tool_service,
            change_tracker=self._change_tracker,
            ingest_run_sink=self._ingest_run_sink,
            scratch=self._config.scratch,
            plugin_name=plugin.metadata.name,
            run_id=self._config.run_id,
        )


def execute_recipe(  # noqa: PLR0913
    recipe: IngestRecipe,
    *,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    tools: ToolsConfig,
    code_profile: ScanProfile,
    config_profile: ScanProfile,
    tool_runner: ToolRunner | None = None,
    tool_service: ToolService | None = None,
    change_tracker: ChangeTracker | None = None,
    ingest_run_sink: IngestRunSink | None = None,
    registry: IngestPluginRegistry | None = None,
) -> RecipeExecutionResult:
    """Execute a recipe with the given configuration.

    Parameters
    ----------
    recipe
        Recipe to execute.
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
        Optional change tracker.
    ingest_run_sink
        Optional run sink.
    registry
        Plugin registry (defaults to global).

    Returns
    -------
    RecipeExecutionResult
        Execution result.
    """
    config = ExecutorConfig(
        registry=registry or get_ingest_registry(),
    )

    executor = RecipeExecutor(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        tools=tools,
        code_profile=code_profile,
        config_profile=config_profile,
        tool_runner=tool_runner,
        tool_service=tool_service,
        change_tracker=change_tracker,
        ingest_run_sink=ingest_run_sink,
        config=config,
    )

    return executor.execute(recipe)


__all__ = [
    "ExecutorConfig",
    "PluginExecutionRecord",
    "RecipeExecutor",
    "execute_recipe",
]
