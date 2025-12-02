"""Graph recipe DSL and executor.

This package provides a declarative recipe system for composing
graph construction and analysis pipelines, similar to the ingestion
recipe system.

Key Components
--------------
- GraphRecipe: Declarative recipe definition
- GraphStage: Stage within a recipe
- RecipeExecutor: Executes recipes
- Builtin recipes: full, incremental, metrics_only, validation_only
"""

from codeintel.graphs.recipes.builtins import (
    BUILDERS_ONLY_RECIPE,
    BUILTIN_RECIPES,
    CALLGRAPH_ONLY_RECIPE,
    FULL_GRAPH_RECIPE,
    IMPORT_GRAPH_ONLY_RECIPE,
    INCREMENTAL_RECIPE,
    METRICS_ONLY_RECIPE,
    get_builtin_recipe,
    list_builtin_recipes,
)
from codeintel.graphs.recipes.dsl import (
    GraphRecipe,
    GraphRecipeOptions,
    GraphStage,
    graph_recipe,
    graph_stage,
)
from codeintel.graphs.recipes.executor import (
    RecipeExecutionResult,
    RecipeExecutor,
    RecipeExecutorContext,
    StageExecutionResult,
    execute_graph_recipe,
)

__all__ = [
    "BUILDERS_ONLY_RECIPE",
    "BUILTIN_RECIPES",
    "CALLGRAPH_ONLY_RECIPE",
    "FULL_GRAPH_RECIPE",
    "IMPORT_GRAPH_ONLY_RECIPE",
    "INCREMENTAL_RECIPE",
    "METRICS_ONLY_RECIPE",
    "GraphRecipe",
    "GraphRecipeOptions",
    "GraphStage",
    "RecipeExecutionResult",
    "RecipeExecutor",
    "RecipeExecutorContext",
    "StageExecutionResult",
    "execute_graph_recipe",
    "get_builtin_recipe",
    "graph_recipe",
    "graph_stage",
    "list_builtin_recipes",
]
