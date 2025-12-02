"""Graph recipe executor.

This module provides the executor for running graph recipes, orchestrating
plugin execution across stages with support for parallelism and failure handling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from codeintel.graphs.core.context import GraphExecutionContext, GraphRuntimeScratch
from codeintel.graphs.core.result import GraphPluginRunRecord
from codeintel.graphs.recipes.dsl import GraphRecipe, GraphStage

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
    records: tuple[GraphPluginRunRecord, ...]
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
    started_at: str
    ended_at: str

    @property
    def all_records(self) -> tuple[GraphPluginRunRecord, ...]:
        """Return all plugin records across all stages.

        Returns
        -------
        tuple[GraphPluginRunRecord, ...]
            All plugin execution records.
        """
        records: list[GraphPluginRunRecord] = []
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
    force_sequential
        Force sequential execution even for parallel stages. Useful when the
        gateway connection is not thread-safe (e.g., shared in-memory DuckDB).
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    engine: GraphEngine | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    force_sequential: bool = False


class RecipeExecutor:
    """Execute graph recipes.

    Orchestrates plugin execution across recipe stages with support for
    parallelism, failure handling, and shared scratch space.
    """

    def __init__(self, context: RecipeExecutorContext) -> None:
        """Initialize the executor.

        Parameters
        ----------
        context
            Execution context.
        """
        self._context = context
        self._scratch = GraphRuntimeScratch()

    def execute(self, recipe: GraphRecipe) -> RecipeExecutionResult:
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
        run_id = uuid4().hex
        start = time.perf_counter()
        started_at = datetime.now(tz=UTC)

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

        ended_at = datetime.now(tz=UTC)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        result = RecipeExecutionResult(
            recipe_name=recipe.name,
            run_id=run_id,
            stages=tuple(stage_results),
            success=overall_success,
            duration_ms=duration_ms,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
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
        stage: GraphStage,
        recipe: GraphRecipe,  # noqa: ARG002 - Reserved for recipe-level options
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
            "recipe_executor.stage.start stage=%s plugins=%s parallel=%s",
            stage.name,
            stage.plugins,
            stage.parallel,
        )

        # Resolve plugins
        from codeintel.graphs.core.registry import get_graph_registry  # noqa: PLC0415

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
        stage: GraphStage,
        run_id: str,
    ) -> list[GraphPluginRunRecord]:
        """Execute plugins within a stage.

        Parameters
        ----------
        plugins
            Plugins to execute.
        stage
            Parent stage.
        run_id
            Run identifier.

        Returns
        -------
        list[GraphPluginRunRecord]
            Execution records.
        """
        if stage.parallel and len(plugins) > 1 and not self._context.force_sequential:
            return self._execute_plugins_parallel(
                plugins=plugins,
                run_id=run_id,
                max_workers=4,  # Default max parallelism
            )
        return self._execute_plugins_sequential(
            plugins=plugins,
            stage=stage,
            run_id=run_id,
        )

    def _execute_plugins_sequential(
        self,
        *,
        plugins: Sequence[GraphPluginProtocol],
        stage: GraphStage,
        run_id: str,
    ) -> list[GraphPluginRunRecord]:
        """Execute plugins sequentially.

        Parameters
        ----------
        plugins
            Plugins to execute.
        stage
            Parent stage.
        run_id
            Run identifier.

        Returns
        -------
        list[GraphPluginRunRecord]
            Execution records.
        """
        records: list[GraphPluginRunRecord] = []

        for plugin in plugins:
            record = self._execute_single_plugin(
                plugin=plugin,
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
        run_id: str,
        max_workers: int,
    ) -> list[GraphPluginRunRecord]:
        """Execute plugins in parallel using a thread pool.

        Parameters
        ----------
        plugins
            Plugins to execute.
        run_id
            Run identifier.
        max_workers
            Maximum number of concurrent workers.

        Returns
        -------
        list[GraphPluginRunRecord]
            Execution records in order of completion.
        """
        records: list[GraphPluginRunRecord] = []

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
                except Exception as exc:
                    log.exception(
                        "recipe_executor.parallel.exception plugin=%s",
                        plugin.metadata.name,
                    )
                    # Create a failure record for the exception
                    records.append(
                        GraphPluginRunRecord(
                            name=plugin.metadata.name,
                            status="failed",
                            started_at=datetime.now(tz=UTC).isoformat(),
                            ended_at=datetime.now(tz=UTC).isoformat(),
                            duration_ms=0.0,
                            attempts=1,
                            partial=True,
                            error=str(exc),
                            meta={},
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
        run_id: str,
    ) -> GraphPluginRunRecord:
        """Execute a single plugin.

        Parameters
        ----------
        plugin
            Plugin to execute.
        run_id
            Run identifier.

        Returns
        -------
        GraphPluginRunRecord
            Execution record.
        """
        start = time.perf_counter()
        started_at = datetime.now(tz=UTC)

        log.info(
            "recipe_executor.plugin.start plugin=%s repo=%s commit=%s",
            plugin.metadata.name,
            self._context.snapshot.repo,
            self._context.snapshot.commit,
        )

        # Build resources from executor context
        from codeintel.graphs.resources.container import ResourceContainer  # noqa: PLC0415
        from codeintel.graphs.resources.graphs import GraphResource  # noqa: PLC0415
        from codeintel.graphs.resources.storage import StorageResource  # noqa: PLC0415

        container = ResourceContainer()
        container.register(StorageResource(self._context.gateway, self._context.snapshot.repo_root))
        if self._context.engine is not None:
            from typing import cast  # noqa: PLC0415

            from codeintel.graphs.engine import NxGraphEngine  # noqa: PLC0415

            container.register(GraphResource(cast("NxGraphEngine", self._context.engine)))

        ctx = GraphExecutionContext(
            snapshot=self._context.snapshot,
            resources=container,
            _gateway=self._context.gateway,
            _engine=self._context.engine,
            _catalog_provider=self._context.catalog_provider,
            scratch=self._scratch,
            plugin_name=plugin.metadata.name,
            run_id=run_id,
        )

        try:
            result = plugin.execute(ctx)
            status = "succeeded" if result.success else "failed"
            error = result.error
        except Exception as exc:
            log.exception(
                "recipe_executor.plugin.exception plugin=%s",
                plugin.metadata.name,
            )
            status = "failed"
            error = str(exc)
            result = None

        ended_at = datetime.now(tz=UTC)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        log.info(
            "recipe_executor.plugin.complete plugin=%s status=%s duration_ms=%.2f",
            plugin.metadata.name,
            status,
            duration_ms,
        )

        return GraphPluginRunRecord(
            name=plugin.metadata.name,
            status=status,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            duration_ms=duration_ms,
            attempts=1,
            partial=status == "failed",
            error=error,
            meta={
                "row_counts": dict(result.row_counts) if result and result.row_counts else None,
            },
        )


def execute_graph_recipe(
    recipe: GraphRecipe,
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
