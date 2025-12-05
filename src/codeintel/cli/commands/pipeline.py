"""Extended pipeline orchestration commands for the CodeIntel CLI.

This module provides additional Typer commands for pipeline management,
including step introspection, dependency visualization, and pipeline execution.

Commands
--------
- **run**: Run the full pipeline using spec-based execution
- **list-steps**: List all available pipeline steps
- **deps**: Show dependency tree for a step
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Annotated

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
    build_runtime_or_exit,
    parse_scope_args,
    setup_logging,
)
from codeintel.graphs.engine.backend import maybe_enable_nx_gpu
from codeintel.pipeline import FULL_PIPELINE, run_pipeline
from codeintel.pipeline.cli_adapter import CliPipelineArgs
from codeintel.pipeline.config_resolver import resolve_scan_profiles, resolve_tools_config
from codeintel.pipeline.steps import REGISTRY, StepPhase
from codeintel.storage.gateway import StorageConfig
from codeintel.storage.gateway_cache import close_gateways, get_gateway

LOG = logging.getLogger(__name__)

pipeline_ext_app = typer.Typer(
    name="pipeline",
    help="Pipeline orchestration commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

SkipScipOpt = Annotated[
    bool,
    typer.Option(
        "--skip-scip",
        is_flag=True,
        help="Skip SCIP ingestion.",
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
    skip_scip: SkipScipOpt = False,
    scope_paths: ScopePathOpt = None,
    scope_modules: ScopeModuleOpt = None,
    scope_time_start: ScopeTimeWindowStartOpt = None,
    scope_time_end: ScopeTimeWindowEndOpt = None,
    nx_gpu: NxGpuOpt = False,
    nx_backend: NxBackendOpt = "auto",
    nx_gpu_strict: NxGpuStrictOpt = False,
    verbose: VerboseOpt = 0,
) -> None:
    r"""Run the full pipeline.

    Execute the complete ingestion, graphs, and analytics pipeline using
    the unified spec-based execution system.

    Examples
    --------
    .. code-block:: bash

        # Run full pipeline
        codeintel pipeline run --repo my-org/repo --commit abc123

        # With scope filtering
        codeintel pipeline run --repo my-org/repo --commit abc123 \
            --scope-path src/core --scope-module core
    """
    setup_logging(verbose)

    # Build runtime using consolidated infrastructure from _common.py
    runtime = build_runtime_or_exit(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
        document_output_dir=document_output_dir,
        nx_gpu=nx_gpu,
        nx_backend=nx_backend,
        nx_gpu_strict=nx_gpu_strict,
    )
    cfg = runtime.cfg

    maybe_enable_nx_gpu(cfg.graph_backend)

    if document_output_dir is not None:
        os.environ.setdefault("CODEINTEL_OUTPUT_DIR", str(document_output_dir))
    if skip_scip:
        os.environ["CODEINTEL_SKIP_SCIP"] = "true"

    try:
        graph_scope = parse_scope_args(scope_paths, scope_modules, scope_time_start, scope_time_end)
    except ValueError as exc:
        typer.secho(f"Invalid scope arguments: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    LOG.info("Running pipeline for repo=%s commit=%s", cfg.repo.repo, cfg.repo.commit)

    profiles = resolve_scan_profiles(repo_root=cfg.paths.repo_root)
    tools = resolve_tools_config()
    cli_args = CliPipelineArgs(
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=cfg.paths.db_path,
        build_dir=cfg.paths.build_dir,
        tools=tools,
        code_profile=profiles.code,
        config_profile=profiles.config,
        graph_backend=cfg.graph_backend,
        graph_scope=graph_scope,
    )

    gateway = get_gateway(StorageConfig.for_ingest(cli_args.db_path))
    try:
        run_pipeline(spec=FULL_PIPELINE, options=cli_args.to_plan_options(gateway, tools))
    finally:
        close_gateways()

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
