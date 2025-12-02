"""Built-in ingestion recipes for common use cases.

This module provides pre-defined recipes for full ingestion,
incremental ingestion, and minimal/quick ingestion scenarios.
"""

from __future__ import annotations

from codeintel.ingestion.recipes.dsl import (
    IngestRecipe,
    RecipeOptions,
    RecipeSpec,
    StageSpec,
    recipe,
    stage,
)

# Full Python ingestion recipe
FULL_PYTHON_RECIPE = recipe(
    name="full_python",
    stages=[
        stage(
            name="scan",
            plugins=["repo_scan"],
            spec=StageSpec(description="Scan repository structure and build module index."),
        ),
        stage(
            name="parse",
            plugins=["ast_extract", "cst_extract"],
            spec=StageSpec(parallel=True, description="Parse Python AST and CST structures."),
        ),
        stage(
            name="index",
            plugins=["scip_ingest"],
            spec=StageSpec(description="Generate SCIP index for semantic analysis."),
        ),
        stage(
            name="enrich",
            plugins=[
                "typing_ingest",
                "coverage_ingest",
                "tests_ingest",
                "docstrings_ingest",
                "config_ingest",
            ],
            spec=StageSpec(
                parallel=True, description="Enrich with typing, coverage, tests, and docstrings."
            ),
        ),
    ],
    spec=RecipeSpec(
        description="Complete Python repository ingestion with all plugins.",
        version="1.0.0",
        options=RecipeOptions(
            enable_incremental=True,
            enable_contracts=True,
            max_parallel_plugins=4,
            fail_fast=True,
        ),
        tags=("python", "full", "default"),
    ),
)

# Incremental ingestion recipe (optimized for repeated runs)
INCREMENTAL_RECIPE = recipe(
    name="incremental",
    stages=[
        stage(
            name="scan",
            plugins=["repo_scan"],
            spec=StageSpec(description="Scan for changed modules."),
        ),
        stage(
            name="parse",
            plugins=["ast_extract", "cst_extract"],
            spec=StageSpec(parallel=True, description="Parse changed modules."),
        ),
        stage(
            name="enrich",
            plugins=["typing_ingest", "docstrings_ingest"],
            spec=StageSpec(parallel=True, description="Update typing and docstrings for changed modules."),
        ),
    ],
    spec=RecipeSpec(
        description="Incremental ingestion for changed files only.",
        version="1.0.0",
        options=RecipeOptions(
            enable_incremental=True,
            enable_contracts=False,
            max_parallel_plugins=4,
            fail_fast=False,
        ),
        tags=("python", "incremental", "fast"),
    ),
)

# Minimal recipe for quick analysis
MINIMAL_RECIPE = recipe(
    name="minimal",
    stages=[
        stage(
            name="scan",
            plugins=["repo_scan"],
            spec=StageSpec(description="Scan repository structure."),
        ),
        stage(
            name="parse",
            plugins=["ast_extract"],
            spec=StageSpec(description="Parse AST for basic analysis."),
        ),
    ],
    spec=RecipeSpec(
        description="Minimal ingestion for quick analysis.",
        version="1.0.0",
        options=RecipeOptions(
            enable_incremental=False,
            enable_contracts=False,
            max_parallel_plugins=2,
            fail_fast=True,
        ),
        tags=("python", "minimal", "quick"),
    ),
)

# Core-only recipe (no external tools required)
CORE_ONLY_RECIPE = recipe(
    name="core_only",
    stages=[
        stage(
            name="scan",
            plugins=["repo_scan"],
            spec=StageSpec(description="Scan repository structure."),
        ),
        stage(
            name="parse",
            plugins=["ast_extract", "cst_extract"],
            spec=StageSpec(parallel=True, description="Parse AST and CST."),
        ),
        stage(
            name="enrich",
            plugins=["docstrings_ingest", "config_ingest"],
            spec=StageSpec(parallel=True, description="Extract docstrings and config."),
        ),
    ],
    spec=RecipeSpec(
        description="Core ingestion without external tool dependencies.",
        version="1.0.0",
        options=RecipeOptions(
            enable_incremental=True,
            enable_contracts=True,
            max_parallel_plugins=4,
            fail_fast=True,
        ),
        disabled_plugins=("typing_ingest", "coverage_ingest", "tests_ingest", "scip_ingest"),
        tags=("python", "core", "no-tools"),
    ),
)

# Analysis-only recipe (skip indexing)
ANALYSIS_RECIPE = recipe(
    name="analysis",
    stages=[
        stage(
            name="scan",
            plugins=["repo_scan"],
            spec=StageSpec(description="Scan repository structure."),
        ),
        stage(
            name="parse",
            plugins=["ast_extract", "cst_extract"],
            spec=StageSpec(parallel=True, description="Parse AST and CST."),
        ),
        stage(
            name="enrich",
            plugins=[
                "typing_ingest",
                "docstrings_ingest",
            ],
            spec=StageSpec(parallel=True, description="Enrich with typing and docstrings."),
        ),
    ],
    spec=RecipeSpec(
        description="Analysis-focused ingestion without SCIP indexing.",
        version="1.0.0",
        options=RecipeOptions(
            enable_incremental=True,
            enable_contracts=True,
            max_parallel_plugins=4,
            fail_fast=True,
        ),
        disabled_plugins=("scip_ingest",),
        tags=("python", "analysis"),
    ),
)

# Registry of built-in recipes
BUILTIN_RECIPES: dict[str, IngestRecipe] = {
    "full_python": FULL_PYTHON_RECIPE,
    "incremental": INCREMENTAL_RECIPE,
    "minimal": MINIMAL_RECIPE,
    "core_only": CORE_ONLY_RECIPE,
    "analysis": ANALYSIS_RECIPE,
}

# Default recipe name
DEFAULT_RECIPE_NAME = "full_python"


def get_builtin_recipe(name: str) -> IngestRecipe | None:
    """Get a built-in recipe by name.

    Parameters
    ----------
    name
        Recipe name.

    Returns
    -------
    IngestRecipe | None
        Recipe if found, None otherwise.
    """
    return BUILTIN_RECIPES.get(name)


def list_builtin_recipes() -> tuple[str, ...]:
    """List all built-in recipe names.

    Returns
    -------
    tuple[str, ...]
        Recipe names.
    """
    return tuple(BUILTIN_RECIPES.keys())


def get_default_recipe() -> IngestRecipe:
    """Get the default recipe.

    Returns
    -------
    IngestRecipe
        Default recipe for full Python ingestion.
    """
    return FULL_PYTHON_RECIPE


__all__ = [
    "ANALYSIS_RECIPE",
    "BUILTIN_RECIPES",
    "CORE_ONLY_RECIPE",
    "DEFAULT_RECIPE_NAME",
    "FULL_PYTHON_RECIPE",
    "INCREMENTAL_RECIPE",
    "MINIMAL_RECIPE",
    "get_builtin_recipe",
    "get_default_recipe",
    "list_builtin_recipes",
]
