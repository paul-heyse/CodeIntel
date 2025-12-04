"""Graph recipe DSL for declarative pipeline composition.

This module re-exports unified recipe types from codeintel.core.recipes,
providing backward compatibility for existing graph recipe code.

The canonical definitions now live in codeintel.core.recipes.
"""

from __future__ import annotations

from codeintel.core.recipes import Recipe, RecipeOptions, RecipeStage, recipe, stage

# Backward-compatible aliases for graph-specific naming
GraphStage = RecipeStage
"""Alias for RecipeStage for backward compatibility."""

GraphRecipeOptions = RecipeOptions
"""Alias for RecipeOptions for backward compatibility."""

GraphRecipe = Recipe
"""Alias for Recipe for backward compatibility."""


def graph_stage(
    name: str,
    plugins: list[str],
    *,
    parallel: bool = False,
    fail_fast: bool = True,
    optional: bool = False,
) -> RecipeStage:
    """Create a graph stage.

    Parameters
    ----------
    name
        Stage identifier.
    plugins
        Plugin names.
    parallel
        Whether plugins can run in parallel.
    fail_fast
        Whether to abort on first failure.
    optional
        Whether the stage can be skipped.

    Returns
    -------
    RecipeStage
        Stage definition.
    """
    return stage(
        name=name,
        plugins=plugins,
        parallel=parallel,
        fail_fast=fail_fast,
        optional=optional,
    )


def graph_recipe(
    name: str,
    *,
    description: str = "",
    stages: list[RecipeStage],
    options: RecipeOptions | None = None,
    version: str = "1.0",
) -> Recipe:
    """Create a graph recipe.

    Parameters
    ----------
    name
        Recipe identifier.
    description
        Human-readable description.
    stages
        Ordered stages to execute.
    options
        Global recipe options.
    version
        Recipe version string.

    Returns
    -------
    Recipe
        Recipe definition.
    """
    return recipe(
        name=name,
        description=description,
        stages=stages,
        options=options,
        version=version,
    )


__all__ = [
    # Graph-specific backward-compatible aliases
    "GraphRecipe",
    "GraphRecipeOptions",
    "GraphStage",
    # Canonical names (from core.recipes)
    "Recipe",
    "RecipeOptions",
    "RecipeStage",
    "graph_recipe",
    "graph_stage",
    "recipe",
    "stage",
]
