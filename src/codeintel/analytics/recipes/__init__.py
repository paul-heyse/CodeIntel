"""Composable analytics recipe system.

This module provides the recipe abstraction for defining and executing
composable analytics workflows. Recipes are named collections of plugins
with configuration overrides that can be executed as a unit.
"""

from __future__ import annotations

from codeintel.analytics.recipes.builtins import (
    COVERAGE_FOCUS,
    FULL_ANALYSIS,
    GRAPH_METRICS,
    QUICK_AUDIT,
    RISK_ANALYSIS,
    TEST_ANALYSIS,
)
from codeintel.analytics.recipes.dsl import RecipeBuilder, recipe
from codeintel.analytics.recipes.executor import (
    RecipeExecutionContext,
    RecipeExecutionPlan,
    RecipeExecutor,
    execute_recipe,
)
from codeintel.analytics.recipes.model import (
    AnalyticsRecipe,
    RecipeExecutionReport,
    RecipePluginRecord,
    RecipeScope,
)
from codeintel.analytics.recipes.registry import (
    RecipeRegistry,
    get_recipe_registry,
    register_recipe,
)

__all__ = [
    "COVERAGE_FOCUS",
    "FULL_ANALYSIS",
    "GRAPH_METRICS",
    "QUICK_AUDIT",
    "RISK_ANALYSIS",
    "TEST_ANALYSIS",
    "AnalyticsRecipe",
    "RecipeBuilder",
    "RecipeExecutionContext",
    "RecipeExecutionPlan",
    "RecipeExecutionReport",
    "RecipeExecutor",
    "RecipePluginRecord",
    "RecipeRegistry",
    "RecipeScope",
    "execute_recipe",
    "get_recipe_registry",
    "recipe",
    "register_recipe",
]
