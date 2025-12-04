"""Unified recipe model definitions.

This module defines the core dataclasses for recipes that work with both
graph, analytics, and ingestion plugins. Base classes provide common
structure that can be extended by domain-specific implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# =============================================================================
# Base Classes (for domain extension)
# =============================================================================


@dataclass(frozen=True)
class BaseRecipeStage:
    """Base stage definition with common fields.

    This class provides the foundational structure for recipe stages
    that can be extended by domain-specific implementations.

    Attributes
    ----------
    name
        Stage identifier.
    plugins
        Plugin names to execute in this stage.
    parallel
        Whether plugins within the stage can run in parallel.
    fail_fast
        Whether to abort stage on first plugin failure.
    """

    name: str
    plugins: tuple[str, ...]
    parallel: bool = False
    fail_fast: bool = True


@dataclass(frozen=True)
class BaseRecipeOptions:
    """Base options with common execution parameters.

    This class provides the foundational structure for recipe options
    that can be extended by domain-specific implementations.

    Attributes
    ----------
    dry_run
        Whether to simulate execution without side effects.
    max_parallel
        Maximum concurrent plugin executions.
    fail_fast
        Whether to stop on first plugin failure.
    """

    dry_run: bool = False
    max_parallel: int = 4
    fail_fast: bool = True


@dataclass(frozen=True)
class BaseRecipe:
    """Base recipe with common structure.

    This class provides the foundational structure for recipes
    that can be extended by domain-specific implementations.

    Attributes
    ----------
    name
        Unique identifier for this recipe.
    description
        Human-readable description.
    version
        Recipe version for cache invalidation.
    tags
        Free-form tags for categorization.
    """

    name: str
    description: str = ""
    version: str = "1.0"
    tags: tuple[str, ...] = ()


# =============================================================================
# Core Recipe Types (extend base classes)
# =============================================================================


@dataclass(frozen=True)
class RecipeStage(BaseRecipeStage):
    """Stage within a recipe.

    Stages allow grouping plugins for ordered execution with
    optional parallelism within a stage.

    Attributes
    ----------
    name
        Stage identifier.
    plugins
        Plugin names to execute in this stage.
    parallel
        Whether plugins within the stage can run in parallel.
    fail_fast
        Whether to abort stage on first plugin failure.
    optional
        Whether the stage can be skipped.
    """

    optional: bool = False


@dataclass(frozen=True)
class RecipeOptions(BaseRecipeOptions):
    """Global options for recipe execution.

    Attributes
    ----------
    dry_run
        Whether to simulate execution without side effects.
    skip_on_unchanged
        Whether to skip plugins when inputs are unchanged.
    max_parallel
        Maximum concurrent plugin executions.
    timeout_ms
        Default timeout per plugin in milliseconds.
    fail_fast
        Whether to stop on first plugin failure.
    max_duration_ms
        Maximum total recipe execution time.
    """

    skip_on_unchanged: bool = False
    timeout_ms: int | None = None
    max_duration_ms: int | None = None


@dataclass(frozen=True)
class RecipeScope:
    """Scope constraints for recipe execution.

    Attributes
    ----------
    paths
        Limit analysis to specific file paths.
    modules
        Limit analysis to specific modules.
    time_window
        Limit analysis to a time range (start, end).
    labels
        Additional labels for filtering.
    """

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe(BaseRecipe):
    """Unified recipe definition for composable workflows.

    A recipe defines an ordered collection of plugins (optionally grouped
    into stages) to execute together, with configuration overrides and
    execution parameters.

    Attributes
    ----------
    name
        Unique identifier for this recipe.
    description
        Human-readable description.
    stages
        Ordered stages to execute (for stage-based recipes).
    plugins
        Flat list of plugins (for simple recipes without stages).
    options
        Global execution options.
    default_configs
        Configuration overrides keyed by plugin name.
    tags
        Free-form tags for categorization.
    version
        Recipe version for cache invalidation.
    """

    stages: tuple[RecipeStage, ...] = ()
    plugins: tuple[str, ...] = ()
    options: RecipeOptions = field(default_factory=RecipeOptions)
    default_configs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    @property
    def all_plugins(self) -> tuple[str, ...]:
        """Return all plugin names across all stages and plugins.

        Returns
        -------
        tuple[str, ...]
            Unique plugin names in execution order.
        """
        plugins: list[str] = []

        # Collect from stages first
        for rec_stage in self.stages:
            for plugin in rec_stage.plugins:
                if plugin not in plugins:
                    plugins.append(plugin)

        # Then collect from flat plugin list
        for plugin in self.plugins:
            if plugin not in plugins:
                plugins.append(plugin)

        return tuple(plugins)

    @property
    def is_staged(self) -> bool:
        """Check if recipe uses stage-based execution.

        Returns
        -------
        bool
            True if stages are defined.
        """
        return len(self.stages) > 0

    def with_plugins(self, *new_plugins: str) -> Recipe:
        """Return a new recipe with additional plugins.

        Parameters
        ----------
        new_plugins
            Plugin names to add.

        Returns
        -------
        Recipe
            New recipe with extended plugin list.
        """
        return Recipe(
            name=self.name,
            description=self.description,
            stages=self.stages,
            plugins=(*self.plugins, *new_plugins),
            options=self.options,
            default_configs=self.default_configs,
            tags=self.tags,
            version=self.version,
        )

    def with_config(
        self,
        plugin_name: str,
        config: Mapping[str, object],
    ) -> Recipe:
        """Return a new recipe with config override for a plugin.

        Parameters
        ----------
        plugin_name
            Plugin to configure.
        config
            Configuration overrides.

        Returns
        -------
        Recipe
            New recipe with updated config.
        """
        new_configs = dict(self.default_configs)
        existing = dict(new_configs.get(plugin_name, {}))
        existing.update(config)
        new_configs[plugin_name] = existing
        return Recipe(
            name=self.name,
            description=self.description,
            stages=self.stages,
            plugins=self.plugins,
            options=self.options,
            default_configs=new_configs,
            tags=self.tags,
            version=self.version,
        )

    def with_options(
        self,
        *,
        dry_run: bool | None = None,
        skip_on_unchanged: bool | None = None,
        max_parallel: int | None = None,
        timeout_ms: int | None = None,
        fail_fast: bool | None = None,
        max_duration_ms: int | None = None,
    ) -> Recipe:
        """Return a new recipe with updated options.

        Parameters
        ----------
        dry_run
            Whether to simulate execution.
        skip_on_unchanged
            Whether to skip unchanged plugins.
        max_parallel
            Maximum concurrent plugin executions.
        timeout_ms
            Default timeout per plugin.
        fail_fast
            Whether to stop on first failure.
        max_duration_ms
            Maximum total execution time.

        Returns
        -------
        Recipe
            New recipe with updated options.
        """
        new_options = RecipeOptions(
            dry_run=dry_run if dry_run is not None else self.options.dry_run,
            skip_on_unchanged=(
                skip_on_unchanged
                if skip_on_unchanged is not None
                else self.options.skip_on_unchanged
            ),
            max_parallel=max_parallel if max_parallel is not None else self.options.max_parallel,
            timeout_ms=timeout_ms if timeout_ms is not None else self.options.timeout_ms,
            fail_fast=fail_fast if fail_fast is not None else self.options.fail_fast,
            max_duration_ms=(
                max_duration_ms if max_duration_ms is not None else self.options.max_duration_ms
            ),
        )
        return Recipe(
            name=self.name,
            description=self.description,
            stages=self.stages,
            plugins=self.plugins,
            options=new_options,
            default_configs=self.default_configs,
            tags=self.tags,
            version=self.version,
        )


@dataclass(frozen=True)
class RecipePluginRecord:
    """Record of a single plugin execution within a recipe.

    Attributes
    ----------
    plugin_name
        Name of the executed plugin.
    status
        Execution status.
    started_at
        When execution started.
    ended_at
        When execution ended.
    duration_ms
        Execution duration in milliseconds.
    attempts
        Number of execution attempts.
    error
        Error message if failed.
    row_counts
        Table row counts from execution.
    meta
        Additional metadata.
    """

    plugin_name: str
    status: Literal["succeeded", "failed", "skipped"]
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attempts: int = 1
    error: str | None = None
    row_counts: Mapping[str, int] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeExecutionReport:
    """Complete report of recipe execution.

    Attributes
    ----------
    recipe_name
        Name of the executed recipe.
    run_id
        Unique run identifier.
    repo
        Repository identifier.
    commit
        Commit identifier.
    scope
        Execution scope.
    started_at
        When execution started.
    ended_at
        When execution ended.
    duration_ms
        Total execution duration.
    status
        Overall execution status.
    plugin_records
        Records for each plugin execution.
    skipped_plugins
        Plugins that were skipped.
    error
        Overall error message if failed.
    tags
        Tags from recipe and runtime.
    """

    recipe_name: str
    run_id: str
    repo: str
    commit: str
    scope: RecipeScope
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    status: Literal["succeeded", "failed", "partial"]
    plugin_records: tuple[RecipePluginRecord, ...]
    skipped_plugins: tuple[str, ...] = ()
    error: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def succeeded_count(self) -> int:
        """Return count of succeeded plugins.

        Returns
        -------
        int
            Number of successful plugin executions.
        """
        return sum(1 for r in self.plugin_records if r.status == "succeeded")

    @property
    def failed_count(self) -> int:
        """Return count of failed plugins.

        Returns
        -------
        int
            Number of failed plugin executions.
        """
        return sum(1 for r in self.plugin_records if r.status == "failed")

    @property
    def total_row_counts(self) -> dict[str, int]:
        """Return aggregated row counts across all plugins.

        Returns
        -------
        dict[str, int]
            Table names to total row counts.
        """
        totals: dict[str, int] = {}
        for record in self.plugin_records:
            for table, count in record.row_counts.items():
                totals[table] = totals.get(table, 0) + count
        return totals


__all__ = [
    # Base classes (for domain extension)
    "BaseRecipe",
    "BaseRecipeOptions",
    "BaseRecipeStage",
    # Core types
    "Recipe",
    "RecipeExecutionReport",
    "RecipeOptions",
    "RecipePluginRecord",
    "RecipeScope",
    "RecipeStage",
]
