"""Unified recipe infrastructure for graphs, analytics, and ingestion.

This package provides a single recipe DSL used by the graphs, analytics,
and ingestion subsystems, eliminating DSL duplication. Base classes are
provided for domain-specific extensions.

Modules
-------
- model: Unified recipe model (Recipe, RecipeStage, RecipeOptions) and base classes
- dsl: Fluent builder and helper functions for recipe construction
- executor: Base recipe executor with scratch space management
"""

from __future__ import annotations

from codeintel.core.recipes.dsl import (
    RecipeBuilder,
    recipe,
    stage,
)
from codeintel.core.recipes.executor import BaseRecipeExecutor
from codeintel.core.recipes.model import (
    BaseRecipe,
    BaseRecipeOptions,
    BaseRecipeStage,
    Recipe,
    RecipeExecutionReport,
    RecipeOptions,
    RecipePluginRecord,
    RecipeScope,
    RecipeStage,
)

__all__ = [
    # Base classes (for domain extension)
    "BaseRecipe",
    "BaseRecipeExecutor",
    "BaseRecipeOptions",
    "BaseRecipeStage",
    # Core types
    "Recipe",
    "RecipeBuilder",
    "RecipeExecutionReport",
    "RecipeOptions",
    "RecipePluginRecord",
    "RecipeScope",
    "RecipeStage",
    "recipe",
    "stage",
]
