"""Recipe model definitions for composable analytics workflows.

This module re-exports unified recipe types from codeintel.core.recipes.
The canonical definitions live in codeintel.core.recipes.
"""

from __future__ import annotations

from codeintel.core.recipes import (
    Recipe,
    RecipeExecutionReport,
    RecipeOptions,
    RecipePluginRecord,
    RecipeScope,
    RecipeStage,
)

__all__ = [
    "Recipe",
    "RecipeExecutionReport",
    "RecipeOptions",
    "RecipePluginRecord",
    "RecipeScope",
    "RecipeStage",
]
