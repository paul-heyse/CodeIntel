"""Recipe model definitions for composable analytics workflows.

This module re-exports unified recipe types from codeintel.core.recipes,
while providing backward-compatible analytics-specific types.

The canonical definitions now live in codeintel.core.recipes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from codeintel.core.recipes import (
    Recipe,
    RecipeExecutionReport,
    RecipeOptions,
    RecipePluginRecord,
    RecipeScope,
    RecipeStage,
)


@dataclass(frozen=True)
class AnalyticsRecipe:
    """Composable analytics workflow definition.

    This is a backward-compatible wrapper around the unified Recipe type,
    providing the analytics-specific flat plugin list API.

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

    def to_unified_recipe(self) -> Recipe:
        """Convert to the unified Recipe type.

        Returns
        -------
        Recipe
            Unified recipe with plugins in a single stage.
        """
        return Recipe(
            name=self.name,
            description=self.description,
            plugins=self.plugins,
            options=RecipeOptions(
                fail_fast=self.fail_fast,
                max_duration_ms=self.max_duration_ms,
            ),
            default_configs=self.default_configs,
            tags=self.tags,
            version=self.version,
        )


__all__ = [
    "AnalyticsRecipe",
    # Re-export unified types for migration
    "Recipe",
    "RecipeExecutionReport",
    "RecipeOptions",
    "RecipePluginRecord",
    "RecipeScope",
    "RecipeStage",
]
