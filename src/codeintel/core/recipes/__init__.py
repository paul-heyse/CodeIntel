"""Unified recipe infrastructure for graphs and analytics.

This package provides a single recipe DSL used by both the graphs
and analytics subsystems, eliminating DSL duplication.

Modules
-------
- model: Unified recipe model (Recipe, RecipeStage, RecipeOptions)
- dsl: Fluent builder and helper functions for recipe construction
- executor: Unified recipe executor
"""

from __future__ import annotations

from codeintel.core.recipes.dsl import (
    RecipeBuilder,
    recipe,
    stage,
)
from codeintel.core.recipes.model import (
    Recipe,
    RecipeExecutionReport,
    RecipeOptions,
    RecipePluginRecord,
    RecipeScope,
    RecipeStage,
)

__all__ = [
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
