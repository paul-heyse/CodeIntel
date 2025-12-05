"""Graph recipe executor.

This module provides the executor for running graph recipes, orchestrating
plugin execution across stages with support for parallelism and failure handling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, cast

from codeintel.config.steps_graphs import GraphPluginPolicy
from codeintel.core.plugins.context import PluginScratch
from codeintel.core.plugins.result import PluginExecutionRecord
from codeintel.core.recipes import Recipe, RecipeStage
from codeintel.core.resources import ResourceRegistry
from codeintel.core.runtime.timing import utc_now
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.registry import get_graph_registry
from codeintel.graphs.engine import NxGraphEngine
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.storage import StorageResource
from codeintel.runtime.ids import new_run_id

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.graphs.core.protocol import GraphPluginProtocol
    from codeintel.graphs.engine import GraphEngine
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageExecutionResult:
    """Result of executing a single stage.

    Attributes
    ----------
    stage_name
        Name of the stage.
    records
        Execution records for each plugin.
    success
        Whether the stage succeeded.
    duration_ms
        Stage duration in milliseconds.
    """

    stage_name: str
    records: tuple[PluginExecutionRecord, ...]
    success: bool
    duration_ms: float


@dataclass(frozen=True)
class RecipeExecutionResult:
    """Result of executing a complete recipe.

    Attributes
    ----------
    recipe_name
        Name of the recipe.
    run_id
        Unique run identifier.
    stages
        Results for each stage.
    success
        Whether the recipe succeeded.
    duration_ms
        Total duration in milliseconds.
    started_at
        Run start time.
    ended_at
        Run end time.
    """

    recipe_name: str
    run_id: str
    stages: tuple[StageExecutionResult, ...]
    success: bool
    duration_ms: float
    started_at: datetime
    ended_at: datetime

    @property
    def all_records(self) -> tuple[PluginExecutionRecord, ...]:
        """Return all plugin records across all stages.

        Returns
        -------
        tuple[PluginExecutionRecord, ...]
            All plugin execution records.
        """
        records: list[PluginExecutionRecord] = []
        for stage in self.stages:
            records.extend(stage.records)
        return tuple(records)

    @property
    def success_count(self) -> int:
        """Return count of successful plugin executions.

        Returns
        -------
        int
            Count of successful plugins.
        """
        return sum(1 for r in self.all_records if r.status == "succeeded")

    @property
    def failure_count(self) -> int:
        """Return count of failed plugin executions.

        Returns
        -------
        int
            Count of failed plugins.
        """
        return sum(1 for r in self.all_records if r.status == "failed")

    @property
    def skip_count(self) -> int:
        """Return count of skipped plugin executions.

        Returns
        -------
        int
            Count of skipped plugins.
        """
        return sum(1 for r in self.all_records if r.status == "skipped")


@dataclass
class RecipeExecutorContext:
    """Context for recipe execution.

    Attributes
    ----------
    gateway
        Storage gateway.
    snapshot
        Repository snapshot.
    engine
        Graph engine.
    catalog_provider
        Function catalog provider.
    policy
        Optional plugin execution policy for stages.
    force_sequential
        Force sequential execution even for parallel stages. Useful when the
        gateway connection is not thread-safe (e.g., shared in-memory DuckDB).
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    engine: GraphEngine | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    policy: GraphPluginPolicy = field(default_factory=GraphPluginPolicy)
    force_sequential: bool = False


class RecipeExecutor:
    """Execute graph recipes with stage-based parallelism.

    Orchestrate plugin execution across recipe stages with support for
    parallelism, failure handling, and shared scratch space.

    Architecture Note
    -----------------
    This executor follows the patterns from BaseRecipeExecutor
    (codeintel.core.recipes.executor) but does not formally extend it because:

    1. Graphs uses stage-based execution (not plan-based)
    2. Graphs has tight integration with GraphPluginExecutionContext
    3. Graphs uses core PluginExecutionRecord directly

    Common patterns shared with other recipe executors:
    - Scratch space management via PluginScratch
    - Parallel execution via ThreadPoolExecutor
    - Stage-based result aggregation
    """

    def __init__(self, context: RecipeExecutorContext) -> None:
        """Initialize the executor.

        Parameters
        ----------
        context
            Execution context.
        """
        self._context = context
        self._scratch = PluginScratch()

    def execute(self, recipe: Recipe) -> RecipeExecutionResult:
        """Execute a graph recipe.

        Parameters
        ----------
        recipe
            Recipe to execute.

        Returns
        -------
        RecipeExecutionResult
            Execution result.
        """
        run_id = new_run_id("graphs")
        start = time.perf_counter()
        started_at = utc_now()

        log.info(
            "recipe_executor.start recipe=%s run_id=%s stages=%d",
            recipe.name,
            run_id,
            len(recipe.stages),
        )

        stage_results: list[StageExecutionResult] = []
        overall_success = True

        try:
            for stage in recipe.stages:
                stage_result = self._execute_stage(
                    stage=stage,
                    recipe=recipe,
                    run_id=run_id,
                )
                stage_results.append(stage_result)

                if not stage_result.success and stage.fail_fast:
                    overall_success = False
                    log.warning(
                        "recipe_executor.stage_failed recipe=%s stage=%s",
                        recipe.name,
                        stage.name,
                    )
                    break
        finally:
            self._scratch.cleanup()

        ended_at = utc_now()
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        result = RecipeExecutionResult(
            recipe_name=recipe.name,
            run_id=run_id,
            stages=tuple(stage_results),
            success=overall_success,
            duration_ms=duration_ms,
            started_at=started_at,
            ended_at=ended_at,
        )

        log.info(
            "recipe_executor.complete recipe=%s run_id=%s success=%s "
            "duration_ms=%.2f succeeded=%d failed=%d skipped=%d",
            recipe.name,
            run_id,
            overall_success,
            duration_ms,
            result.success_count,
            result.failure_count,
            result.skip_count,
        )

        return result

    def _execute_stage(
        self,
        *,
        stage: RecipeStage,
        recipe: Recipe,
        run_id: str,
    ) -> StageExecutionResult:
        """Execute a single stage.

        Parameters
        ----------
        stage
            Stage to execute.
        recipe
            Parent recipe.
        run_id
            Run identifier.

        Returns
        -------
        StageExecutionResult
            Stage execution result.
        """
        start = time.perf_counter()

        log.info(
            "recipe_executor.stage.start recipe=%s stage=%s plugins=%s parallel=%s",
            recipe.name,
            stage.name,
            stage.plugins,
            stage.parallel,
        )

        # Resolve plugins
        registry = get_graph_registry()
        plugins: list[GraphPluginProtocol] = []

        for plugin_name in stage.plugins:
            try:
                plugin = registry.get(plugin_name)
                plugins.append(plugin)
            except KeyError:
                log.warning(
                    "recipe_executor.plugin_not_found stage=%s plugin=%s",
                    stage.name,
                    plugin_name,
                )

        # Execute plugins
        records = self._execute_plugins(
            plugins=plugins,
            stage=stage,
            recipe=recipe,
            run_id=run_id,
        )

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        success = all(r.status != "failed" for r in records)

        log.info(
            "recipe_executor.stage.complete stage=%s success=%s duration_ms=%.2f",
            stage.name,
            success,
            duration_ms,
        )

        return StageExecutionResult(
            stage_name=stage.name,
            records=tuple(records),
            success=success,
            duration_ms=duration_ms,
        )

    def _execute_plugins(
        self,
        *,
        plugins: Sequence[GraphPluginProtocol],
        stage: RecipeStage,
        recipe: Recipe,
        run_id: str,
    ) -> list[PluginExecutionRecord]:
        """Execute plugins within a stage.

        Parameters
        ----------
        plugins
            Plugins to execute.
        stage
            Parent stage.
        recipe
            Parent recipe.
        run_id
            Run identifier.

        Returns
        -------
        list[PluginExecutionRecord]
            Execution records.
        """
        if stage.parallel and len(plugins) > 1 and not self._context.force_sequential:
            return self._execute_plugins_parallel(
                plugins=plugins,
                stage=stage,
                recipe=recipe,
                run_id=run_id,
                max_workers=4,  # Default max parallelism
            )
        return self._execute_plugins_sequential(
            plugins=plugins,
            stage=stage,
            recipe=recipe,
            run_id=run_id,
        )

    def _execute_plugins_sequential(
        self,
        *,
        plugins: Sequence[GraphPluginProtocol],
        stage: RecipeStage,
        recipe: Recipe,
        run_id: str,
    ) -> list[PluginExecutionRecord]:
        """Execute plugins sequentially.

        Parameters
        ----------
        plugins
            Plugins to execute.
        stage
            Parent stage.
        recipe
            Parent recipe.
        run_id
            Run identifier.

        Returns
        -------
        list[PluginExecutionRecord]
            Execution records.
        """
        records: list[PluginExecutionRecord] = []

        for plugin in plugins:
            record = self._execute_single_plugin(
                plugin=plugin,
                stage_name=stage.name,
                recipe_name=recipe.name,
                run_id=run_id,
            )
            records.append(record)

            if record.status == "failed" and stage.fail_fast:
                break

        return records

    def _execute_plugins_parallel(
        self,
        *,
        plugins: Sequence[GraphPluginProtocol],
        stage: RecipeStage,
        recipe: Recipe,
        run_id: str,
        max_workers: int,
    ) -> list[PluginExecutionRecord]:
        """Execute plugins in parallel using a thread pool.

        Parameters
        ----------
        plugins
            Plugins to execute.
        stage
            Parent stage.
        recipe
            Parent recipe.
        run_id
            Run identifier.
        max_workers
            Maximum number of concurrent workers.

        Returns
        -------
        list[PluginExecutionRecord]
            Execution records in order of completion.
        """
        records: list[PluginExecutionRecord] = []

        log.info(
            "recipe_executor.parallel.start plugins=%d max_workers=%d",
            len(plugins),
            max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all plugins
            future_to_plugin = {
                executor.submit(
                    self._execute_single_plugin,
                    plugin=plugin,
                    stage_name=stage.name,
                    recipe_name=recipe.name,
                    run_id=run_id,
                ): plugin
                for plugin in plugins
            }

            # Collect results as they complete
            for future in as_completed(future_to_plugin):
                plugin = future_to_plugin[future]
                try:
                    record = future.result()
                    records.append(record)
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    LookupError,
                    OSError,
                ) as exc:
                    log.exception(
                        "recipe_executor.parallel.exception plugin=%s",
                        plugin.metadata.name,
                    )
                    # Create a failure record for the exception
                    now = utc_now()
                    records.append(
                        PluginExecutionRecord(
                            plugin_name=plugin.metadata.name,
                            status="failed",
                            started_at=now,
                            ended_at=now,
                            duration_ms=0.0,
                            attempts=1,
                            partial=True,
                            error=str(exc),
                            meta={
                                "recipe": recipe.name,
                                "stage": stage.name,
                            },
                        )
                    )

        log.info(
            "recipe_executor.parallel.complete plugins=%d succeeded=%d failed=%d",
            len(plugins),
            sum(1 for r in records if r.status == "succeeded"),
            sum(1 for r in records if r.status == "failed"),
        )

        return records

    def _execute_single_plugin(
        self,
        *,
        plugin: GraphPluginProtocol,
        stage_name: str,
        recipe_name: str,
        run_id: str,
    ) -> PluginExecutionRecord:
        """Execute a single plugin.

        Parameters
        ----------
        plugin
            Plugin to execute.
        stage_name
            Name of the stage the plugin belongs to.
        recipe_name
            Name of the recipe being executed.
        run_id
            Run identifier.

        Returns
        -------
        PluginExecutionRecord
            Execution record.
        """
        start = time.perf_counter()
        started_at = utc_now()

        log.info(
            "recipe_executor.plugin.start recipe=%s stage=%s plugin=%s repo=%s commit=%s",
            recipe_name,
            stage_name,
            plugin.metadata.name,
            self._context.snapshot.repo,
            self._context.snapshot.commit,
        )

        # Build resources from executor context
        resources = ResourceRegistry()
        resources.register(
            StorageResource,
            StorageResource(self._context.gateway, self._context.snapshot.repo_root),
        )
        if self._context.engine is not None:
            resources.register(
                GraphResource, GraphResource(cast("NxGraphEngine", self._context.engine))
            )

        ctx = GraphPluginExecutionContext(
            gateway=self._context.gateway,
            snapshot=self._context.snapshot,
            run_id=run_id,
            resources=resources,
            scratch=self._scratch,
            plugin_name=plugin.metadata.name,
            _catalog_provider=self._context.catalog_provider,
        )

        try:
            result = plugin.execute(ctx)
            status = "succeeded" if result.success else "failed"
            error = result.error
        except (
            RuntimeError,
            ValueError,
            TypeError,
            LookupError,
            OSError,
        ) as exc:
            log.exception(
                "recipe_executor.plugin.exception plugin=%s",
                plugin.metadata.name,
            )
            status = "failed"
            error = str(exc)
            result = None

        ended_at = utc_now()
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        log.info(
            (
                "recipe_executor.plugin.complete recipe=%s stage=%s plugin=%s "
                "status=%s duration_ms=%.2f"
            ),
            recipe_name,
            stage_name,
            plugin.metadata.name,
            status,
            duration_ms,
        )

        return PluginExecutionRecord(
            plugin_name=plugin.metadata.name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            attempts=1,
            partial=status == "failed",
            result=result,  # Store actual result for row_counts access
            error=error,
            meta={
                "recipe": recipe_name,
                "stage": stage_name,
            },
        )


def execute_graph_recipe(
    recipe: Recipe,
    *,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    engine: GraphEngine | None = None,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> RecipeExecutionResult:
    """Execute a graph recipe.

    Parameters
    ----------
    recipe
        Recipe to execute.
    gateway
        Storage gateway.
    snapshot
        Repository snapshot.
    engine
        Graph engine.
    catalog_provider
        Function catalog provider.

    Returns
    -------
    RecipeExecutionResult
        Execution result.
    """
    context = RecipeExecutorContext(
        gateway=gateway,
        snapshot=snapshot,
        engine=engine,
        catalog_provider=catalog_provider,
    )
    executor = RecipeExecutor(context)
    return executor.execute(recipe)


__all__ = [
    "RecipeExecutionResult",
    "RecipeExecutor",
    "RecipeExecutorContext",
    "StageExecutionResult",
    "execute_graph_recipe",
]
