"""Recipe model definitions for composable analytics workflows.

This module defines the core dataclasses for analytics recipes,
including the recipe itself and execution reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


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
        Limit analysis to a time range.
    labels
        Additional labels for filtering.
    """

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsRecipe:
    """Composable analytics workflow definition.

    A recipe defines a named collection of plugins to execute together,
    with optional configuration overrides and execution parameters.

    Attributes
    ----------
    name
        Unique identifier for this recipe.
    description
        Human-readable description of what this recipe does.
    plugins
        Ordered tuple of plugin names to execute.
    default_configs
        Default configuration overrides keyed by plugin name.
    tags
        Free-form tags for categorization.
    fail_fast
        Whether to stop on first failure.
    parallel_stages
        Whether stages can be parallelized.
    max_duration_ms
        Maximum total execution time in milliseconds.
    version
        Recipe version for cache invalidation.
    """

    name: str
    description: str
    plugins: tuple[str, ...]
    default_configs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    fail_fast: bool = True
    parallel_stages: bool = False
    max_duration_ms: int | None = None
    version: str = "1.0.0"

    def with_plugins(self, *plugins: str) -> AnalyticsRecipe:
        """Return a new recipe with additional plugins.

        Parameters
        ----------
        plugins
            Plugin names to add.

        Returns
        -------
        AnalyticsRecipe
            New recipe with extended plugin list.
        """
        return AnalyticsRecipe(
            name=self.name,
            description=self.description,
            plugins=(*self.plugins, *plugins),
            default_configs=self.default_configs,
            tags=self.tags,
            fail_fast=self.fail_fast,
            parallel_stages=self.parallel_stages,
            max_duration_ms=self.max_duration_ms,
            version=self.version,
        )

    def with_config(
        self,
        plugin_name: str,
        config: Mapping[str, object],
    ) -> AnalyticsRecipe:
        """Return a new recipe with config override for a plugin.

        Parameters
        ----------
        plugin_name
            Plugin to configure.
        config
            Configuration overrides.

        Returns
        -------
        AnalyticsRecipe
            New recipe with updated config.
        """
        new_configs = dict(self.default_configs)
        existing = dict(new_configs.get(plugin_name, {}))
        existing.update(config)
        new_configs[plugin_name] = existing
        return AnalyticsRecipe(
            name=self.name,
            description=self.description,
            plugins=self.plugins,
            default_configs=new_configs,
            tags=self.tags,
            fail_fast=self.fail_fast,
            parallel_stages=self.parallel_stages,
            max_duration_ms=self.max_duration_ms,
            version=self.version,
        )

    def with_fail_fast(self, *, fail_fast: bool) -> AnalyticsRecipe:
        """Return a new recipe with updated fail_fast setting.

        Parameters
        ----------
        fail_fast
            New fail_fast value.

        Returns
        -------
        AnalyticsRecipe
            New recipe with updated setting.
        """
        return AnalyticsRecipe(
            name=self.name,
            description=self.description,
            plugins=self.plugins,
            default_configs=self.default_configs,
            tags=self.tags,
            fail_fast=fail_fast,
            parallel_stages=self.parallel_stages,
            max_duration_ms=self.max_duration_ms,
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
        """Return count of succeeded plugins."""
        return sum(1 for r in self.plugin_records if r.status == "succeeded")

    @property
    def failed_count(self) -> int:
        """Return count of failed plugins."""
        return sum(1 for r in self.plugin_records if r.status == "failed")

    @property
    def total_row_counts(self) -> dict[str, int]:
        """Return aggregated row counts across all plugins."""
        totals: dict[str, int] = {}
        for record in self.plugin_records:
            for table, count in record.row_counts.items():
                totals[table] = totals.get(table, 0) + count
        return totals


__all__ = [
    "AnalyticsRecipe",
    "RecipeExecutionReport",
    "RecipePluginRecord",
    "RecipeScope",
]
