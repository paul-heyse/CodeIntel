"""Fluent DSL for building analytics recipes.

This module provides a builder pattern for constructing analytics
recipes with a clean, chainable API.

The canonical recipe builder now lives in codeintel.core.recipes.dsl.
This module re-exports it and provides analytics-specific convenience.
"""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.analytics.recipes.model import AnalyticsRecipe
from codeintel.core.recipes import RecipeBuilder as CoreRecipeBuilder
from codeintel.core.recipes import recipe as core_recipe
from codeintel.core.recipes import stage as core_stage


class RecipeBuilder:
    """Fluent builder for constructing analytics recipes.

    Provides a clean API for building recipes incrementally.

    Example
    -------
    >>> recipe = (
    ...     RecipeBuilder("custom_analysis")
    ...     .description("Custom analysis workflow")
    ...     .add("functions.metrics")
    ...     .add("hotspots.build")
    ...     .with_config("hotspots.build", {"max_commits": 500})
    ...     .tag("custom")
    ...     .fail_fast(True)
    ...     .build()
    ... )
    """

    def __init__(self, name: str) -> None:
        """Initialize a recipe builder.

        Parameters
        ----------
        name
            Name for the recipe being built.
        """
        self._name = name
        self._description = ""
        self._plugins: list[str] = []
        self._configs: dict[str, dict[str, object]] = {}
        self._tags: list[str] = []
        self._fail_fast = True
        self._parallel_stages = False
        self._max_duration_ms: int | None = None
        self._version = "1.0.0"

    def description(self, desc: str) -> RecipeBuilder:
        """Set the recipe description.

        Parameters
        ----------
        desc
            Human-readable description.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._description = desc
        return self

    def add(self, plugin_name: str) -> RecipeBuilder:
        """Add a plugin to the recipe.

        Parameters
        ----------
        plugin_name
            Name of the plugin to add.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        if plugin_name not in self._plugins:
            self._plugins.append(plugin_name)
        return self

    def add_all(self, *plugin_names: str) -> RecipeBuilder:
        """Add multiple plugins to the recipe.

        Parameters
        ----------
        plugin_names
            Names of plugins to add.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        for name in plugin_names:
            self.add(name)
        return self

    def remove(self, plugin_name: str) -> RecipeBuilder:
        """Remove a plugin from the recipe.

        Parameters
        ----------
        plugin_name
            Name of the plugin to remove.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        if plugin_name in self._plugins:
            self._plugins.remove(plugin_name)
        return self

    def with_config(
        self,
        plugin_name: str,
        config: Mapping[str, object],
    ) -> RecipeBuilder:
        """Set configuration for a specific plugin.

        Parameters
        ----------
        plugin_name
            Plugin to configure.
        config
            Configuration mapping.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        if plugin_name not in self._configs:
            self._configs[plugin_name] = {}
        self._configs[plugin_name].update(config)
        return self

    def tag(self, *tags: str) -> RecipeBuilder:
        """Add tags to the recipe.

        Parameters
        ----------
        tags
            Tags to add.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        for t in tags:
            if t not in self._tags:
                self._tags.append(t)
        return self

    def fail_fast(self, *, value: bool) -> RecipeBuilder:
        """Set the fail_fast behavior.

        Parameters
        ----------
        value
            Whether to stop on first failure.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._fail_fast = value
        return self

    def parallel_stages(self, *, value: bool) -> RecipeBuilder:
        """Set whether stages can be parallelized.

        Parameters
        ----------
        value
            Whether to allow parallel execution.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._parallel_stages = value
        return self

    def max_duration(self, ms: int | None) -> RecipeBuilder:
        """Set maximum execution duration.

        Parameters
        ----------
        ms
            Maximum duration in milliseconds.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._max_duration_ms = ms
        return self

    def version(self, v: str) -> RecipeBuilder:
        """Set the recipe version.

        Parameters
        ----------
        v
            Version string.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._version = v
        return self

    def extend(self, recipe: AnalyticsRecipe) -> RecipeBuilder:
        """Extend this recipe with plugins from another recipe.

        Parameters
        ----------
        recipe
            Recipe to extend from.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        for plugin in recipe.plugins:
            self.add(plugin)
        for plugin_name, config in recipe.default_configs.items():
            self.with_config(plugin_name, config)
        for t in recipe.tags:
            self.tag(t)
        return self

    def build(self) -> AnalyticsRecipe:
        """Build the recipe.

        Returns
        -------
        AnalyticsRecipe
            The constructed recipe.
        """
        return AnalyticsRecipe(
            name=self._name,
            description=self._description,
            plugins=tuple(self._plugins),
            default_configs=dict(self._configs),
            tags=tuple(self._tags),
            fail_fast=self._fail_fast,
            parallel_stages=self._parallel_stages,
            max_duration_ms=self._max_duration_ms,
            version=self._version,
        )


def recipe(name: str) -> RecipeBuilder:
    """Start building a new recipe.

    Parameters
    ----------
    name
        Name for the recipe.

    Returns
    -------
    RecipeBuilder
        A new recipe builder.

    Example
    -------
    >>> my_recipe = (
    ...     recipe("my_analysis").description("My custom analysis").add("functions.metrics").build()
    ... )
    """
    return RecipeBuilder(name)


__all__ = [
    # Re-export core types for gradual migration
    "CoreRecipeBuilder",
    # Analytics-specific types
    "RecipeBuilder",
    "core_recipe",
    "core_stage",
    "recipe",
]
