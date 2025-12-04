"""Extended pipeline orchestration commands for the CodeIntel CLI.

This module provides additional Typer commands for pipeline management,
including step introspection, dependency visualization, and Prefect flow
execution.

Commands
--------
- **run**: Run the full pipeline via Prefect
- **list-steps**: List all available pipeline steps
- **deps**: Show dependency tree for a step
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    DocumentOutputDirOpt,
    JsonOutputOpt,
    NxBackendOpt,
    NxGpuOpt,
    NxGpuStrictOpt,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    ScopeModuleOpt,
    ScopePathOpt,
    ScopeTimeWindowEndOpt,
    ScopeTimeWindowStartOpt,
    VerboseOpt,
    build_graph_backend_config,
    build_graph_feature_flags_from_env,
    parse_scope_args,
    setup_logging,
)
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.parser_types import FunctionParserKind
from codeintel.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.ingestion.utilities.scanning import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.pipeline.orchestration.prefect_flow import ExportArgs, export_docs_flow
from codeintel.pipeline.orchestration.steps import REGISTRY, StepPhase

LOG = logging.getLogger(__name__)

pipeline_ext_app = typer.Typer(
    name="pipeline",
    help="Pipeline orchestration commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

TargetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--target",
        help="Name of a pipeline step to run (can be repeated).",
    ),
]

SkipScipOpt = Annotated[
    bool,
    typer.Option(
        "--skip-scip",
        is_flag=True,
        help="Skip SCIP ingestion.",
    ),
]

FunctionFailOnMissingSpansOpt = Annotated[
    bool,
    typer.Option(
        "--function-fail-on-missing-spans",
        is_flag=True,
        help="Fail pipeline when function spans are missing.",
    ),
]

FunctionParserOpt = Annotated[
    str | None,
    typer.Option(
        "--function-parser",
        help="Parser selector for function analytics (e.g., 'python').",
    ),
]

HistoryCommitOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--history-commit",
        help="Commit SHA to include in history_timeseries (can be repeated).",
    ),
]

HistoryDbDirOpt = Annotated[
    Path,
    typer.Option(
        "--history-db-dir",
        help="Directory containing per-commit DuckDB snapshots.",
    ),
]

ExportDatasetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--export-dataset",
        help="Dataset name to export during docs export step (can be repeated).",
    ),
]

ExportValidationProfileOpt = Annotated[
    str | None,
    typer.Option(
        "--export-validation-profile",
        help="Override validation profile: strict or lenient.",
    ),
]

ForceFullExportOpt = Annotated[
    bool,
    typer.Option(
        "--force-full-export",
        is_flag=True,
        help="Force re-export even when incremental markers match.",
    ),
]

PhaseFilterOpt = Annotated[
    str | None,
    typer.Option(
        "--phase",
        help="Filter steps by phase: ingestion, graphs, analytics, or export.",
    ),
]

StepNameArg = Annotated[
    str,
    typer.Argument(help="Name of the step to show dependencies for."),
]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@pipeline_ext_app.command("run")
def pipeline_run(
    project_root: ProjectRootOpt = None,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    document_output_dir: DocumentOutputDirOpt = None,
    targets: TargetOpt = None,
    skip_scip: SkipScipOpt = False,
    function_fail_on_missing_spans: FunctionFailOnMissingSpansOpt = False,
    function_parser: FunctionParserOpt = None,
    history_commits: HistoryCommitOpt = None,
    history_db_dir: HistoryDbDirOpt = Path("build/db"),
    export_datasets: ExportDatasetOpt = None,
    export_validation_profile: ExportValidationProfileOpt = None,
    force_full_export: ForceFullExportOpt = False,
    scope_paths: ScopePathOpt = None,
    scope_modules: ScopeModuleOpt = None,
    scope_time_start: ScopeTimeWindowStartOpt = None,
    scope_time_end: ScopeTimeWindowEndOpt = None,
    nx_gpu: NxGpuOpt = False,
    nx_backend: NxBackendOpt = "auto",
    nx_gpu_strict: NxGpuStrictOpt = False,
    verbose: VerboseOpt = 0,
) -> None:
    r"""Run the full pipeline via Prefect.

    Executes the complete ingestion, graphs, analytics, and export pipeline
    with optional targeting and scope filtering.

    Examples
    --------
    .. code-block:: bash

        # Run full pipeline
        codeintel pipeline run --repo my-org/repo --commit abc123

        # Run specific targets
        codeintel pipeline run --repo my-org/repo --commit abc123 \
            --target export_docs

        # With scope filtering
        codeintel pipeline run --repo my-org/repo --commit abc123 \
            --scope-path src/core --scope-module core
    """
    setup_logging(verbose)

    # Build configuration
    from codeintel.cli.project import ProjectNotFoundError, find_project_root, load_project_config

    try:
        project_root_path = find_project_root(project_root)
        project_config = load_project_config(project_root_path)

        resolved_repo = repo or project_config.repo
        from codeintel.cli.project import detect_commit

        resolved_commit = commit or detect_commit(project_root_path)
        resolved_db_path = db_path or (project_root_path / project_config.storage.db_path)
        resolved_repo_root = repo_root or project_root_path
        resolved_build_dir = build_dir or (project_root_path / ".codeintel")
    except ProjectNotFoundError:
        if repo is None or commit is None:
            typer.secho(
                "Error: No codeintel.yaml found. Provide --repo and --commit explicitly.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from None
        resolved_repo = repo
        resolved_commit = commit
        resolved_db_path = db_path or Path("build/db/codeintel_prefect.duckdb")
        resolved_repo_root = repo_root or Path.cwd()
        resolved_build_dir = build_dir or Path("build")

    graph_backend = build_graph_backend_config(nx_gpu, nx_backend, nx_gpu_strict)
    graph_features = build_graph_feature_flags_from_env()

    paths_cfg = CliPathsInput(
        repo_root=resolved_repo_root,
        build_dir=resolved_build_dir,
        db_path=resolved_db_path,
        document_output_dir=document_output_dir,
    )
    repo_cfg = RepoConfig(repo=resolved_repo, commit=resolved_commit)
    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(graph_backend=graph_backend, graph_features=graph_features),
    )

    maybe_enable_nx_gpu(cfg.graph_backend)

    if document_output_dir is not None:
        os.environ.setdefault("CODEINTEL_OUTPUT_DIR", str(document_output_dir))

    if skip_scip:
        os.environ["CODEINTEL_SKIP_SCIP"] = "true"

    target_list = list(targets) if targets else None

    try:
        graph_scope = parse_scope_args(scope_paths, scope_modules, scope_time_start, scope_time_end)
    except ValueError as exc:
        typer.secho(f"Invalid scope arguments: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    LOG.info(
        "Running Prefect export_docs_flow for repo=%s commit=%s targets=%s",
        resolved_repo,
        resolved_commit,
        target_list,
    )

    from codeintel.config.models import ToolsConfig

    tools = ToolsConfig.default()

    export_docs_flow(
        args=ExportArgs(
            repo_root=cfg.paths.repo_root,
            repo=resolved_repo,
            commit=resolved_commit,
            db_path=cfg.paths.db_path,
            build_dir=cfg.paths.build_dir,
            tools=tools,
            code_profile=profile_from_env(default_code_profile(cfg.paths.repo_root)),
            config_profile=profile_from_env(default_config_profile(cfg.paths.repo_root)),
            function_fail_on_missing_spans=function_fail_on_missing_spans,
            function_parser=FunctionParserKind(function_parser) if function_parser else None,
            history_commits=tuple(history_commits) if history_commits else None,
            history_db_dir=history_db_dir,
            graph_backend=cfg.graph_backend,
            export_datasets=tuple(export_datasets) if export_datasets else None,
            export_validation_profile=cast(
                "Literal['strict', 'lenient'] | None", export_validation_profile
            ),
            force_full_export=force_full_export,
            graph_scope=graph_scope,
        ),
        targets=target_list,
    )

    typer.secho("Pipeline completed.", fg=typer.colors.GREEN)


@pipeline_ext_app.command("list-steps")
def pipeline_list_steps(
    phase: PhaseFilterOpt = None,
    json_output: JsonOutputOpt = False,
) -> None:
    """List all available pipeline steps with descriptions.

    Shows registered pipeline steps with their phase, description,
    and dependencies.

    Examples
    --------
    .. code-block:: bash

        # List all steps
        codeintel pipeline list-steps

        # Filter by phase
        codeintel pipeline list-steps --phase analytics

        # Output as JSON
        codeintel pipeline list-steps --json
    """
    if phase:
        step_phase = StepPhase(phase)
        steps = REGISTRY.list_by_phase(step_phase)
    else:
        steps = REGISTRY.list_all()

    if json_output:
        data = [
            {
                "name": meta.name,
                "description": meta.description,
                "phase": meta.phase.value,
                "deps": list(meta.deps),
            }
            for meta in steps
        ]
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        for meta in steps:
            deps_str = ", ".join(meta.deps) if meta.deps else "(none)"
            sys.stdout.write(f"{meta.name} [{meta.phase.value}]\n")
            sys.stdout.write(f"  {meta.description}\n")
            sys.stdout.write(f"  deps: {deps_str}\n")
            sys.stdout.write("\n")


@pipeline_ext_app.command("deps")
def pipeline_deps(
    step_name: StepNameArg,
    json_output: JsonOutputOpt = False,
) -> None:
    """Show dependency tree for a pipeline step.

    Displays direct and transitive dependencies for the specified step.

    Examples
    --------
    .. code-block:: bash

        # Show dependencies
        codeintel pipeline deps export_docs

        # Output as JSON
        codeintel pipeline deps export_docs --json
    """
    if step_name not in REGISTRY:
        typer.secho(f"Unknown step: {step_name}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Get all transitive dependencies
    expanded = REGISTRY.expand_with_deps([step_name])
    expanded.discard(step_name)  # Remove the step itself from deps

    # Get direct deps
    direct_deps = tuple(REGISTRY.get_deps(step_name))

    if json_output:
        data = {
            "step": step_name,
            "direct_deps": list(direct_deps),
            "transitive_deps": sorted(expanded),
        }
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        step = REGISTRY[step_name]
        sys.stdout.write(f"Step: {step_name}\n")
        sys.stdout.write(f"Description: {step.description}\n")
        sys.stdout.write(f"Phase: {step.phase.value}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"Direct dependencies ({len(direct_deps)}):\n")
        if direct_deps:
            for dep in direct_deps:
                sys.stdout.write(f"  - {dep}\n")
        else:
            sys.stdout.write("  (none)\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"All transitive dependencies ({len(expanded)}):\n")
        if expanded:
            for dep in sorted(expanded):
                sys.stdout.write(f"  - {dep}\n")
        else:
            sys.stdout.write("  (none)\n")


__all__ = ["pipeline_ext_app"]
