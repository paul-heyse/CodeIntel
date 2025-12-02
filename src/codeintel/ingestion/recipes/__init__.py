"""Recipe DSL and execution engine for ingestion pipelines.

This package provides declarative recipe composition and execution
for ingestion pipelines, with support for parallelism, failure
handling, and extensibility.
"""

from __future__ import annotations

from codeintel.ingestion.recipes.builtin import (
    ANALYSIS_RECIPE,
    BUILTIN_RECIPES,
    CORE_ONLY_RECIPE,
    DEFAULT_RECIPE_NAME,
    FULL_PYTHON_RECIPE,
    INCREMENTAL_RECIPE,
    MINIMAL_RECIPE,
    get_builtin_recipe,
    get_default_recipe,
    list_builtin_recipes,
)
from codeintel.ingestion.recipes.dsl import (
    IngestRecipe,
    RecipeExecutionResult,
    RecipeOptions,
    RecipeSpec,
    RecipeStage,
    RecipeStageResult,
    StageSpec,
    recipe,
    stage,
)
from codeintel.ingestion.recipes.executor import (
    ExecutorConfig,
    PluginExecutionRecord,
    RecipeExecutor,
    execute_recipe,
)

__all__ = [
    "ANALYSIS_RECIPE",
    "BUILTIN_RECIPES",
    "CORE_ONLY_RECIPE",
    "DEFAULT_RECIPE_NAME",
    "FULL_PYTHON_RECIPE",
    "INCREMENTAL_RECIPE",
    "MINIMAL_RECIPE",
    "ExecutorConfig",
    "IngestRecipe",
    "PluginExecutionRecord",
    "RecipeExecutionResult",
    "RecipeExecutor",
    "RecipeOptions",
    "RecipeSpec",
    "RecipeStage",
    "RecipeStageResult",
    "StageSpec",
    "execute_recipe",
    "get_builtin_recipe",
    "get_default_recipe",
    "list_builtin_recipes",
    "recipe",
    "stage",
]
