"""Builtin graph recipes.

This module provides standard graph recipes for common workflows:
- full: All builders + all metrics + validation
- incremental: Changed files only
- builders_only: Just graph construction
- metrics_only: Skip builders, run metrics on existing graphs
- validation_only: Just validation checks
"""

from __future__ import annotations

from codeintel.graphs.recipes.dsl import (
    GraphRecipe,
    GraphRecipeOptions,
    graph_recipe,
    graph_stage,
)

# Full pipeline: Build all graphs and compute all metrics
FULL_GRAPH_RECIPE = graph_recipe(
    name="full",
    description="Build all graphs and compute all metrics.",
    stages=[
        graph_stage(
            "goids",
            ["goid_builder"],
        ),
        graph_stage(
            "structure",
            ["import_graph_builder"],
        ),
        graph_stage(
            "edges",
            ["callgraph_builder", "cfg_dfg_builder"],
        ),
        graph_stage(
            "core_metrics",
            [
                "core_graph_metrics",
                "graph_metrics_functions_ext",
                "graph_metrics_modules_ext",
            ],
            parallel=True,
        ),
        graph_stage(
            "secondary_metrics",
            [
                "cfg_metrics",
                "dfg_metrics",
                "test_graph_metrics",
                "subsystem_graph_metrics",
            ],
            parallel=True,
        ),
        graph_stage(
            "stats",
            ["graph_stats"],
        ),
    ],
)

# Builders only: Just construct graphs without metrics
BUILDERS_ONLY_RECIPE = graph_recipe(
    name="builders_only",
    description="Build all graphs without computing metrics.",
    stages=[
        graph_stage(
            "goids",
            ["goid_builder"],
        ),
        graph_stage(
            "structure",
            ["import_graph_builder"],
        ),
        graph_stage(
            "edges",
            ["callgraph_builder", "cfg_dfg_builder"],
        ),
    ],
)

# Metrics only: Compute metrics on existing graphs
METRICS_ONLY_RECIPE = graph_recipe(
    name="metrics_only",
    description="Compute metrics on existing graphs.",
    stages=[
        graph_stage(
            "core_metrics",
            [
                "core_graph_metrics",
                "graph_metrics_functions_ext",
                "graph_metrics_modules_ext",
            ],
            parallel=True,
        ),
        graph_stage(
            "secondary_metrics",
            [
                "cfg_metrics",
                "dfg_metrics",
                "test_graph_metrics",
                "subsystem_graph_metrics",
            ],
            parallel=True,
        ),
        graph_stage(
            "stats",
            ["graph_stats"],
        ),
    ],
)

# Incremental: Only rebuild changed files
INCREMENTAL_RECIPE = graph_recipe(
    name="incremental",
    description="Incrementally build graphs for changed files.",
    stages=[
        graph_stage(
            "goids",
            ["goid_builder"],
        ),
        graph_stage(
            "structure",
            ["import_graph_builder"],
        ),
        graph_stage(
            "edges",
            ["callgraph_builder", "cfg_dfg_builder"],
        ),
        graph_stage(
            "core_metrics",
            [
                "core_graph_metrics",
                "graph_metrics_functions_ext",
                "graph_metrics_modules_ext",
            ],
            parallel=True,
        ),
    ],
    options=GraphRecipeOptions(skip_on_unchanged=True),
)

# Call graph only: Minimal call graph construction
CALLGRAPH_ONLY_RECIPE = graph_recipe(
    name="callgraph_only",
    description="Build only the call graph.",
    stages=[
        graph_stage(
            "goids",
            ["goid_builder"],
        ),
        graph_stage(
            "edges",
            ["callgraph_builder"],
        ),
    ],
)

# Import graph only: Minimal import graph construction
IMPORT_GRAPH_ONLY_RECIPE = graph_recipe(
    name="import_graph_only",
    description="Build only the import graph.",
    stages=[
        graph_stage(
            "structure",
            ["import_graph_builder"],
        ),
    ],
)


# Registry of builtin recipes
BUILTIN_RECIPES: dict[str, GraphRecipe] = {
    "full": FULL_GRAPH_RECIPE,
    "builders_only": BUILDERS_ONLY_RECIPE,
    "metrics_only": METRICS_ONLY_RECIPE,
    "incremental": INCREMENTAL_RECIPE,
    "callgraph_only": CALLGRAPH_ONLY_RECIPE,
    "import_graph_only": IMPORT_GRAPH_ONLY_RECIPE,
}


def get_builtin_recipe(name: str) -> GraphRecipe:
    """Get a builtin recipe by name.

    Parameters
    ----------
    name
        Recipe name.

    Returns
    -------
    GraphRecipe
        The builtin recipe.

    Raises
    ------
    KeyError
        If no recipe exists with the given name.
    """
    if name not in BUILTIN_RECIPES:
        available = ", ".join(sorted(BUILTIN_RECIPES.keys()))
        message = f"Unknown builtin recipe '{name}'. Available: {available}"
        raise KeyError(message)
    return BUILTIN_RECIPES[name]


def list_builtin_recipes() -> tuple[str, ...]:
    """List all builtin recipe names.

    Returns
    -------
    tuple[str, ...]
        Available builtin recipe names.
    """
    return tuple(sorted(BUILTIN_RECIPES.keys()))


__all__ = [
    "BUILDERS_ONLY_RECIPE",
    "BUILTIN_RECIPES",
    "CALLGRAPH_ONLY_RECIPE",
    "FULL_GRAPH_RECIPE",
    "IMPORT_GRAPH_ONLY_RECIPE",
    "INCREMENTAL_RECIPE",
    "METRICS_ONLY_RECIPE",
    "get_builtin_recipe",
    "list_builtin_recipes",
]
