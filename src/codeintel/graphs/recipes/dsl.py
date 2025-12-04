"""Graph recipe DSL for declarative pipeline composition.

This module provides graph-specific wrapper functions around the unified
recipe types from codeintel.core.recipes.

The canonical types (Recipe, RecipeStage, RecipeOptions) are re-exported
for convenience, while graph_recipe() and graph_stage() provide domain-
specific factory functions.
"""

from __future__ import annotations

from codeintel.core.recipes import Recipe, RecipeOptions, RecipeStage, recipe, stage


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
    "Recipe",
    "RecipeOptions",
    "RecipeStage",
    "graph_recipe",
    "graph_stage",
    "recipe",
    "stage",
]
