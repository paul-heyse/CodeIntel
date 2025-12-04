"""Fluent DSL for building analytics recipes.

This module provides a builder pattern for constructing analytics
recipes with a clean, chainable API.

The canonical recipe builder now lives in codeintel.core.recipes.dsl.
This module extends it and provides analytics-specific convenience.
"""

from __future__ import annotations

from codeintel.analytics.recipes.model import Recipe
from codeintel.core.recipes import RecipeBuilder as CoreRecipeBuilder
from codeintel.core.recipes import recipe as core_recipe
from codeintel.core.recipes import stage as core_stage


class RecipeBuilder(CoreRecipeBuilder):
    """Fluent builder for constructing analytics recipes.

    Extends CoreRecipeBuilder with analytics-specific convenience.

    Example
    -------
    >>> recipe = (
    ...     RecipeBuilder("custom_analysis")
    ...     .description("Custom analysis workflow")
    ...     .add("functions.metrics")
    ...     .add("hotspots.build")
    ...     .with_config("hotspots.build", {"max_commits": 500})
    ...     .tag("custom")
    ...     .fail_fast()
    ...     .build()
    ... )
    """

    def extend(self, other: Recipe) -> RecipeBuilder:
        """Extend this recipe with plugins from another recipe.

        Parameters
        ----------
        other
            Recipe to extend from.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        # Use all_plugins to include both staged and flat plugins
        for plugin in other.all_plugins:
            self.add(plugin)
        for plugin_name, config in other.default_configs.items():
            self.with_config(plugin_name, config)
        for t in other.tags:
            self.tag(t)
        return self


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
