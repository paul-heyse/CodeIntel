"""Recipe execution engine.

This module provides the executor for running analytics recipes,
handling dependency resolution, plugin execution, and telemetry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from codeintel.analytics.core.execution_context import (
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.analytics.core.plugin_protocol import (
    AnalyticsPluginProtocol,
    PluginResult,
)
from codeintel.analytics.core.registry import PluginRegistry, get_registry
from codeintel.analytics.recipes.model import (
    AnalyticsRecipe,
    RecipeExecutionReport,
    RecipePluginRecord,
    RecipeScope,
)
from codeintel.analytics.recipes.registry import RecipeRegistry, get_recipe_registry
from codeintel.analytics.runtime_manifest import AnalyticsScope
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecipeExecutionContext:
    """Context for recipe execution.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Snapshot reference for the analysis.
    scope
        Execution scope constraints.
    config_overrides
        Per-plugin configuration overrides.
    graph_runtime
        Optional graph runtime for graph-aware plugins.
    catalog_provider
        Optional function catalog.
    analytics_context
        Optional analytics context.
    extra
        Additional context values.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    scope: RecipeScope = field(default_factory=RecipeScope)
    config_overrides: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    graph_runtime: GraphRuntime | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    analytics_context: AnalyticsContext | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeExecutionPlan:
    """Execution plan for a recipe.

    Attributes
    ----------
    recipe
        The recipe being executed.
    plugins
        Ordered plugins to execute.
    run_id
        Unique run identifier.
    resolved_configs
        Merged configs for each plugin.
    """

    recipe: AnalyticsRecipe
    plugins: tuple[AnalyticsPluginProtocol, ...]
    run_id: str
    resolved_configs: Mapping[str, Mapping[str, object]]


class RecipeExecutor:
    """Execute analytics recipes with dependency resolution.

    The executor handles:
    - Plugin resolution and ordering
    - Configuration merging
    - Execution with retry and timeout
    - Telemetry and reporting
    """

    def __init__(
        self,
        plugin_registry: PluginRegistry | None = None,
        recipe_registry: RecipeRegistry | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        plugin_registry
            Plugin registry to use. Defaults to global registry.
        recipe_registry
            Recipe registry to use. Defaults to global registry.
        """
        self._plugin_registry = plugin_registry or get_registry()
        self._recipe_registry = recipe_registry or get_recipe_registry()

    def plan(
        self,
        recipe: str | AnalyticsRecipe,
        *,
        config_overrides: Mapping[str, Mapping[str, object]] | None = None,
    ) -> RecipeExecutionPlan:
        """Build an execution plan for a recipe.

        Parameters
        ----------
        recipe
            Recipe name or instance to plan.
        config_overrides
            Additional configuration overrides.

        Returns
        -------
        RecipeExecutionPlan
            Execution plan with resolved plugins.
        """
        resolved_recipe = self._resolve_recipe(recipe)
        plugins = self._resolve_plugins(resolved_recipe)
        merged_configs = self.merge_configs(resolved_recipe, config_overrides)

        return RecipeExecutionPlan(
            recipe=resolved_recipe,
            plugins=plugins,
            run_id=uuid4().hex,
            resolved_configs=merged_configs,
        )

    def execute(
        self,
        recipe: str | AnalyticsRecipe,
        context: RecipeExecutionContext,
        *,
        config_overrides: Mapping[str, Mapping[str, object]] | None = None,
    ) -> RecipeExecutionReport:
        """Execute a recipe.

        Parameters
        ----------
        recipe
            Recipe name or instance to execute.
        context
            Execution context with gateway and snapshot.
        config_overrides
            Additional configuration overrides.

        Returns
        -------
        RecipeExecutionReport
            Complete execution report.
        """
        plan = self.plan(recipe, config_overrides=config_overrides)
        return self._execute_plan(plan, context)

    def _resolve_recipe(self, recipe: str | AnalyticsRecipe) -> AnalyticsRecipe:
        """Resolve a recipe reference to an instance.

        Returns
        -------
        AnalyticsRecipe
            Resolved recipe instance.
        """
        if isinstance(recipe, AnalyticsRecipe):
            return recipe
        return self._recipe_registry.get(recipe)

    def _resolve_plugins(
        self,
        recipe: AnalyticsRecipe,
    ) -> tuple[AnalyticsPluginProtocol, ...]:
        """Resolve and order plugins for a recipe.

        Returns
        -------
        tuple[AnalyticsPluginProtocol, ...]
            Ordered plugins specified by the recipe.
        """
        plugin_plan = self._plugin_registry.plan(list(recipe.plugins))
        return plugin_plan.plugins

    @staticmethod
    def merge_configs(
        recipe: AnalyticsRecipe,
        overrides: Mapping[str, Mapping[str, object]] | None,
    ) -> dict[str, dict[str, object]]:
        """Merge recipe default configs with overrides.

        Returns
        -------
        dict[str, dict[str, object]]
            Combined configuration mapping by plugin name.
        """
        merged: dict[str, dict[str, object]] = {}

        # Apply recipe defaults
        for plugin_name, config in recipe.default_configs.items():
            merged[plugin_name] = dict(config)

        # Apply overrides
        if overrides:
            for plugin_name, config in overrides.items():
                if plugin_name not in merged:
                    merged[plugin_name] = {}
                merged[plugin_name].update(config)

        return merged

    def _execute_plan(
        self,
        plan: RecipeExecutionPlan,
        context: RecipeExecutionContext,
    ) -> RecipeExecutionReport:
        """Execute a planned recipe.

        Returns
        -------
        RecipeExecutionReport
            Execution report for the recipe run.
        """
        started_at = datetime.now(tz=UTC)
        start_time = time.perf_counter()

        records: list[RecipePluginRecord] = []
        skipped: list[str] = []
        overall_status: Literal["succeeded", "failed", "partial"] = "succeeded"
        overall_error: str | None = None

        # Shared scratch for plugin communication
        scratch = PluginScratch()

        for plugin in plan.plugins:
            plugin_name = plugin.metadata.name
            plugin_config = plan.resolved_configs.get(plugin_name, {})

            record = self.execute_plugin(
                plugin=plugin,
                context=context,
                config=plugin_config,
                run_id=plan.run_id,
                scratch=scratch,
            )
            records.append(record)

            if record.status == "failed":
                if plan.recipe.fail_fast:
                    overall_status = "failed"
                    overall_error = f"Plugin {plugin_name} failed: {record.error}"
                    # Mark remaining plugins as skipped
                    remaining_idx = plan.plugins.index(plugin) + 1
                    skipped.extend(
                        remaining.metadata.name for remaining in plan.plugins[remaining_idx:]
                    )
                    break
                overall_status = "partial"

        # Cleanup scratch
        scratch.cleanup()

        ended_at = datetime.now(tz=UTC)
        duration_ms = (time.perf_counter() - start_time) * 1000

        return RecipeExecutionReport(
            recipe_name=plan.recipe.name,
            run_id=plan.run_id,
            repo=context.snapshot.repo,
            commit=context.snapshot.commit,
            scope=context.scope,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            status=overall_status,
            plugin_records=tuple(records),
            skipped_plugins=tuple(skipped),
            error=overall_error,
            tags=dict(context.extra.get("tags", {})),
        )

    @staticmethod
    def execute_plugin(
        plugin: AnalyticsPluginProtocol,
        context: RecipeExecutionContext,
        config: Mapping[str, object],
        run_id: str,
        scratch: PluginScratch,
    ) -> RecipePluginRecord:
        """Execute a single plugin.

        Returns
        -------
        RecipePluginRecord
            Record summarizing the plugin execution.
        """
        plugin_name = plugin.metadata.name
        started_at = datetime.now(tz=UTC)
        start_time = time.perf_counter()

        # Build plugin execution context
        builder = PluginExecutionContextBuilder(
            gateway=context.gateway,
            snapshot=context.snapshot,
            run_id=run_id,
            scope=_to_analytics_scope(context.scope),
        )

        if context.graph_runtime is not None:
            builder.with_graph_runtime(context.graph_runtime)
        if context.catalog_provider is not None:
            builder.with_catalog(context.catalog_provider)
        if context.analytics_context is not None:
            builder.with_analytics_context(context.analytics_context)

        builder.with_options(config)
        builder.with_plugin_name(plugin_name)

        for key, value in context.extra.items():
            builder.with_extra(key, value)

        plugin_ctx = builder.build(scratch=scratch)

        # Validate inputs
        validation = plugin.validate_inputs(plugin_ctx)
        if not validation.valid:
            ended_at = datetime.now(tz=UTC)
            duration_ms = (time.perf_counter() - start_time) * 1000
            return RecipePluginRecord(
                plugin_name=plugin_name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                error=f"Validation failed: {', '.join(validation.errors)}",
            )

        # Execute plugin
        try:
            result = plugin.execute(plugin_ctx)
            status: Literal["succeeded", "failed", "skipped"] = (
                "succeeded" if result.success else "failed"
            )
            error = result.error
            row_counts = dict(result.row_counts)
        except Exception as exc:
            log.exception("Plugin %s failed with exception", plugin_name)
            status = "failed"
            error = str(exc)
            row_counts = {}
            result = PluginResult.fail(error)

        ended_at = datetime.now(tz=UTC)
        duration_ms = (time.perf_counter() - start_time) * 1000

        return RecipePluginRecord(
            plugin_name=plugin_name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            error=error,
            row_counts=row_counts,
            meta=dict(result.meta) if result else {},
        )


def _to_analytics_scope(scope: RecipeScope) -> AnalyticsScope:
    """Convert RecipeScope to AnalyticsScope.

    Returns
    -------
    AnalyticsScope
        AnalyticsScope constructed from the recipe scope values.
    """
    return AnalyticsScope(
        paths=scope.paths,
        modules=scope.modules,
        time_window=scope.time_window,
        labels=dict(scope.labels),
    )


@dataclass
class RecipeExecutionOptions:
    """Options for recipe execution convenience function."""

    scope: RecipeScope | None = None
    config_overrides: Mapping[str, Mapping[str, object]] | None = None
    graph_runtime: GraphRuntime | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    analytics_context: AnalyticsContext | None = None


def execute_recipe(
    recipe: str | AnalyticsRecipe,
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    options: RecipeExecutionOptions | None = None,
) -> RecipeExecutionReport:
    """Execute a recipe with minimal boilerplate.

    Parameters
    ----------
    recipe
        Recipe name or instance.
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    options
        Execution options including scope, config overrides, and runtime resources.

    Returns
    -------
    RecipeExecutionReport
        Execution report.
    """
    opts = options or RecipeExecutionOptions()
    context = RecipeExecutionContext(
        gateway=gateway,
        snapshot=snapshot,
        scope=opts.scope or RecipeScope(),
        config_overrides=opts.config_overrides or {},
        graph_runtime=opts.graph_runtime,
        catalog_provider=opts.catalog_provider,
        analytics_context=opts.analytics_context,
    )
    executor = RecipeExecutor()
    return executor.execute(recipe, context, config_overrides=opts.config_overrides)


__all__ = [
    "RecipeExecutionContext",
    "RecipeExecutionOptions",
    "RecipeExecutionPlan",
    "RecipeExecutor",
    "execute_recipe",
]
