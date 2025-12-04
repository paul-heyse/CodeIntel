"""Ingestion plugin commands for the CodeIntel CLI.

This module provides Typer commands for running and managing ingestion plugins,
including recipe-based execution and plugin introspection.

Commands
--------
- **run**: Execute ingestion using a recipe or explicit plugin list
- **plugins**: List registered ingestion plugins with metadata
- **recipes**: List available built-in recipes
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    DocumentOutputDirOpt,
    JsonOutputOpt,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    VerboseOpt,
    build_paths_from_cli,
    build_runtime_or_exit,
    setup_logging,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.ingestion.plugins import (
    DEFAULT_INGEST_PLUGINS,
    list_ingest_plugins,
    plan_ingest_plugins,
)
from codeintel.ingestion.plugins.registry import PlanOptions
from codeintel.ingestion.recipes import (
    BUILTIN_RECIPES,
    IngestRecipe,
    RecipeExecutionResult,
    RecipeOptions,
    RecipeSpec,
    StageSpec,
    execute_recipe_for_context,
    get_builtin_recipe,
    recipe,
    stage,
)
from codeintel.ingestion.recipes.executor import RecipeExecutorContext
from codeintel.ingestion.tools.infrastructure import ToolRunner
from codeintel.ingestion.tools.service import ToolService
from codeintel.ingestion.utilities.scanning import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.runtime import new_run_context

LOG = logging.getLogger(__name__)

ingest_app = typer.Typer(
    name="ingest",
    help="Ingestion plugin orchestration commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

RecipeOpt = Annotated[
    str | None,
    typer.Option(
        "--recipe",
        help="Recipe name to execute (e.g., 'full_python', 'incremental', 'minimal').",
    ),
]

RecipeFileOpt = Annotated[
    Path | None,
    typer.Option(
        "--recipe-file",
        help="Path to a YAML recipe file to execute.",
    ),
]

PluginsOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--plugins",
        help="Explicit plugin names to run (overrides recipe).",
    ),
]

DisableOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--disable",
        help="Plugins to disable from the recipe or default set.",
    ),
]

SkipScipOpt = Annotated[
    bool,
    typer.Option(
        "--skip-scip",
        is_flag=True,
        help="Skip SCIP ingestion (convenience for --disable scip_ingest).",
    ),
]

ParallelOpt = Annotated[
    bool,
    typer.Option(
        "--parallel/--no-parallel",
        help="Enable/disable parallel execution within stages (default: enabled).",
    ),
]

FailFastOpt = Annotated[
    bool,
    typer.Option(
        "--fail-fast/--no-fail-fast",
        help="Stop on first plugin failure (default: enabled).",
    ),
]

PlanOpt = Annotated[
    bool,
    typer.Option(
        "--plan",
        is_flag=True,
        help="Show planned execution order plus dependency graph.",
    ),
]

NamesOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--names",
        help="Explicit plugin names to plan/list (defaults to built-in defaults).",
    ),
]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _resolve_ingest_recipe(
    recipe_name: str | None,
    recipe_file: Path | None,
    plugins: list[str] | None,
    disabled_plugins: list[str] | None,
    parallel: bool,
    fail_fast: bool,
) -> IngestRecipe | None:
    """Resolve ingest recipe from options.

    Parameters
    ----------
    recipe_name
        Built-in recipe name.
    recipe_file
        Path to YAML recipe file.
    plugins
        Explicit plugin names.
    disabled_plugins
        Plugins to disable.
    parallel
        Whether to enable parallel execution.
    fail_fast
        Whether to stop on first failure.

    Returns
    -------
    IngestRecipe | None
        Resolved recipe or None if not found.
    """
    if recipe_file is not None:
        return IngestRecipe.from_yaml(recipe_file)
    if recipe_name is not None:
        ingest_recipe = get_builtin_recipe(recipe_name)
        if ingest_recipe is None:
            LOG.error("Unknown recipe: %s", recipe_name)
        return ingest_recipe
    if plugins:
        plugin_names = tuple(plugins)
        return recipe(
            name="cli_explicit",
            stages=[
                stage(name="run", plugins=plugin_names, spec=StageSpec(parallel=parallel)),
            ],
            spec=RecipeSpec(
                description="CLI-specified plugin list",
                options=RecipeOptions(
                    fail_fast=fail_fast,
                    enable_incremental=True,
                ),
                disabled_plugins=tuple(disabled_plugins or ()),
            ),
        )
    return recipe(
        name="cli_default",
        stages=[
            stage(name="scan", plugins=["repo_scan"]),
            stage(
                name="parse",
                plugins=["ast_extract", "cst_extract"],
                spec=StageSpec(parallel=parallel),
            ),
            stage(name="index", plugins=["scip_ingest"]),
            stage(
                name="enrich",
                plugins=[
                    "typing_ingest",
                    "coverage_ingest",
                    "tests_ingest",
                    "docstrings_ingest",
                    "config_ingest",
                ],
                spec=StageSpec(parallel=parallel),
            ),
        ],
        spec=RecipeSpec(
            description="Default ingestion pipeline",
            options=RecipeOptions(
                fail_fast=fail_fast,
                enable_incremental=True,
            ),
            disabled_plugins=tuple(disabled_plugins or ()),
        ),
    )


def _collect_disabled_plugins(
    disabled: list[str] | None,
    skip_scip: bool,
) -> tuple[str, ...]:
    """Collect disabled plugin names.

    Parameters
    ----------
    disabled
        Explicit disabled plugins.
    skip_scip
        Whether to skip SCIP ingestion.

    Returns
    -------
    tuple[str, ...]
        Disabled plugin names.
    """
    disabled_list: list[str] = list(disabled or [])
    if skip_scip:
        disabled_list.append("scip_ingest")
    return tuple(disabled_list)


def _apply_disabled_to_recipe(
    ingest_recipe: IngestRecipe,
    disabled: tuple[str, ...],
) -> IngestRecipe:
    """Apply disabled plugins to recipe.

    Parameters
    ----------
    ingest_recipe
        Source recipe.
    disabled
        Disabled plugin names.

    Returns
    -------
    IngestRecipe
        Recipe with disabled plugins applied.
    """
    if disabled:
        return ingest_recipe.with_disabled(*disabled)
    return ingest_recipe


def _render_ingest_result(result: RecipeExecutionResult, *, output_json: bool) -> None:
    """Render ingest result to stdout.

    Parameters
    ----------
    result
        Execution result.
    output_json
        Whether to output JSON.
    """
    if output_json:
        payload = {
            "success": result.success,
            "recipe": result.recipe.name,
            "duration_s": result.duration_s,
            "error": result.error,
            "stages": [
                {
                    "name": sr.stage.name,
                    "success": sr.success,
                    "duration_s": sr.duration_s,
                    "plugins": dict(sr.plugin_results),
                }
                for sr in result.stage_results
            ],
            "skipped_stages": list(result.skipped_stages),
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return
    sys.stdout.write(f"Recipe: {result.recipe.name}\n")
    sys.stdout.write(f"Success: {result.success}\n")
    sys.stdout.write(f"Duration: {result.duration_s:.2f}s\n")
    if result.error:
        sys.stdout.write(f"Error: {result.error}\n")
    for sr in result.stage_results:
        sys.stdout.write(f"\nStage: {sr.stage.name} (success={sr.success})\n")
        for plugin_name, plugin_result in sr.plugin_results.items():
            sys.stdout.write(f"  - {plugin_name}: {plugin_result}\n")


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@ingest_app.command("run")
def ingest_run(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    document_output_dir: DocumentOutputDirOpt = None,
    recipe_name: RecipeOpt = None,
    recipe_file: RecipeFileOpt = None,
    plugins: PluginsOpt = None,
    disable: DisableOpt = None,
    skip_scip: SkipScipOpt = False,
    parallel: ParallelOpt = True,
    fail_fast: FailFastOpt = True,
    json_output: JsonOutputOpt = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Run ingestion using a recipe or explicit plugin list.

    Uses codeintel.yaml for configuration if available, otherwise requires
    explicit --repo and --commit options.

    Examples
    --------
    .. code-block:: bash

        # Using project file
        codeintel ingest run

        # Using explicit options
        codeintel ingest run --repo my-org/my-repo --commit abc123

        # Using a specific recipe
        codeintel ingest run --recipe full_python

        # Using explicit plugins
        codeintel ingest run --plugins repo_scan --plugins ast_extract
    """
    setup_logging(verbose)

    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
        document_output_dir=document_output_dir,
    )

    disabled = _collect_disabled_plugins(disable, skip_scip)
    ingest_recipe = _resolve_ingest_recipe(
        recipe_name=recipe_name,
        recipe_file=recipe_file,
        plugins=plugins,
        disabled_plugins=list(disabled) if disabled else None,
        parallel=parallel,
        fail_fast=fail_fast,
    )
    if ingest_recipe is None:
        raise typer.Exit(code=1)

    ingest_recipe = _apply_disabled_to_recipe(ingest_recipe, disabled)

    snapshot = SnapshotRef(
        repo=runtime.project.repo,
        commit=runtime.cfg.repo.commit,
        repo_root=runtime.cfg.paths.repo_root,
    )
    paths = build_paths_from_cli(runtime.cfg.paths)
    code_profile = profile_from_env(default_code_profile(runtime.cfg.paths.repo_root))
    config_profile = profile_from_env(default_config_profile(runtime.cfg.paths.repo_root))

    runner = ToolRunner(
        cache_dir=runtime.cfg.paths.build_dir / ".tool_cache",
        tools_config=runtime.tools,
    )
    tool_service = ToolService(runner, runtime.tools)

    LOG.info(
        "Running ingest recipe=%s plugins=%s",
        ingest_recipe.name,
        ingest_recipe.all_plugins,
    )

    run_ctx = new_run_context(snapshot=snapshot, kind="ingest", trigger="cli")

    context = RecipeExecutorContext(
        gateway=runtime.gateway,
        snapshot=snapshot,
        paths=paths,
        tools=runtime.tools,
        code_profile=code_profile,
        config_profile=config_profile,
        tool_runner=runner,
        tool_service=tool_service,
        run_context=run_ctx,
    )
    result = execute_recipe_for_context(ingest_recipe, run_ctx, context)

    _render_ingest_result(result, output_json=json_output)
    if not result.success:
        raise typer.Exit(code=1)


@ingest_app.command("plugins")
def ingest_plugins(
    plan: PlanOpt = False,
    names: NamesOpt = None,
    disable: DisableOpt = None,
    json_output: JsonOutputOpt = False,
) -> None:
    """List registered ingestion plugins with metadata.

    Shows all registered plugins or a planned execution order when --plan
    is specified.

    Examples
    --------
    .. code-block:: bash

        # List all plugins
        codeintel ingest plugins

        # Show execution plan
        codeintel ingest plugins --plan

        # Output as JSON
        codeintel ingest plugins --json
    """
    requested_names = tuple(names) if names else None
    disabled = tuple(disable) if disable else ()
    requested = requested_names if requested_names is not None else DEFAULT_INGEST_PLUGINS

    if plan:
        try:
            plan_result = plan_ingest_plugins(
                PlanOptions(
                    plugin_names=requested,
                    disabled=disabled,
                    defaults=DEFAULT_INGEST_PLUGINS,
                )
            )
        except ValueError:
            LOG.exception("Invalid ingest plugin plan for names=%s", requested)
            raise typer.Exit(code=1) from None

        if json_output:
            payload = {
                "plan_id": plan_result.plan_id,
                "ordered_plugins": list(plan_result.ordered_names),
                "skipped_plugins": [
                    {"name": skipped.name, "reason": skipped.reason}
                    for skipped in plan_result.skipped_plugins
                ],
                "dep_graph": {name: list(deps) for name, deps in plan_result.dep_graph.items()},
                "plugin_metadata": {
                    plugin.metadata.name: {
                        "stage": plugin.metadata.stage,
                        "severity": plugin.metadata.severity,
                        "depends_on": list(plugin.metadata.depends_on),
                        "provides": list(plugin.metadata.provides),
                        "requires": list(plugin.metadata.requires),
                        "produces_tables": list(plugin.metadata.produces_tables),
                        "tool_dependencies": list(plugin.metadata.tool_dependencies),
                        "supports_incremental": plugin.metadata.supports_incremental,
                        "isolation_kind": plugin.metadata.isolation_kind,
                    }
                    for plugin in plan_result.plugins
                },
            }
            sys.stdout.write(json.dumps(payload, indent=2))
            sys.stdout.write("\n")
        else:
            sys.stdout.write(f"Plan ID: {plan_result.plan_id}\n")
            sys.stdout.write("Execution order:\n")
            for plugin in plan_result.plugins:
                meta = plugin.metadata
                sys.stdout.write(f"  - {meta.name} [{meta.stage} | {meta.severity}]\n")
            if plan_result.skipped_plugins:
                sys.stdout.write("Skipped:\n")
                for skipped in plan_result.skipped_plugins:
                    sys.stdout.write(f"  - {skipped.name} ({skipped.reason})\n")
        return

    plugins_list = list_ingest_plugins()
    if json_output:
        payload = {
            "count": len(plugins_list),
            "plugins": {
                plugin.metadata.name: {
                    "name": plugin.metadata.name,
                    "description": plugin.metadata.description,
                    "stage": plugin.metadata.stage,
                    "severity": plugin.metadata.severity,
                    "enabled_by_default": plugin.metadata.enabled_by_default,
                    "depends_on": list(plugin.metadata.depends_on),
                    "provides": list(plugin.metadata.provides),
                    "requires": list(plugin.metadata.requires),
                    "produces_tables": list(plugin.metadata.produces_tables),
                    "tool_dependencies": list(plugin.metadata.tool_dependencies),
                    "supports_incremental": plugin.metadata.supports_incremental,
                    "isolation_kind": plugin.metadata.isolation_kind,
                }
                for plugin in plugins_list
            },
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

    for plugin in plugins_list:
        meta = plugin.metadata
        sys.stdout.write(f"- {meta.name} [{meta.stage}]\n")
        sys.stdout.write(f"    {meta.description}\n")


@ingest_app.command("recipes")
def ingest_recipes(
    json_output: JsonOutputOpt = False,
) -> None:
    """List available built-in ingestion recipes.

    Shows all registered recipes with their stages and metadata.

    Examples
    --------
    .. code-block:: bash

        # List all recipes
        codeintel ingest recipes

        # Output as JSON
        codeintel ingest recipes --json
    """
    if json_output:
        payload = {
            "count": len(BUILTIN_RECIPES),
            "recipes": {
                name: {
                    "name": recipe_obj.name,
                    "description": recipe_obj.description,
                    "version": recipe_obj.version,
                    "stages": [
                        {
                            "name": recipe_stage.name,
                            "plugins": list(recipe_stage.plugins),
                            "parallel": recipe_stage.parallel,
                        }
                        for recipe_stage in recipe_obj.stages
                    ],
                    "tags": list(recipe_obj.tags),
                }
                for name, recipe_obj in BUILTIN_RECIPES.items()
            },
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

    for name, recipe_obj in BUILTIN_RECIPES.items():
        sys.stdout.write(f"- {name} (v{recipe_obj.version})\n")
        sys.stdout.write(f"    {recipe_obj.description}\n")
        sys.stdout.write(f"    Stages: {', '.join(recipe_obj.stage_names)}\n")
        if recipe_obj.tags:
            sys.stdout.write(f"    Tags: {', '.join(recipe_obj.tags)}\n")


__all__ = ["ingest_app"]
