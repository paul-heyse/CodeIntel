"""Cyclopts wiring for the build command group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import typer
from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormat, RuntimeCliOptions
from codeintel.cli.commands.build import (
    BuildHistoryOptions,
    BuildRunContext,
    BuildRunOptions,
    BuildStatusOptions,
    RunMode,
    TargetScope,
    build_history_handler,
    build_run_handler,
    build_status_handler,
)
from codeintel.cli.cyclopts_common import JsonFlag, OutputFmt, ProjectRoot, Verbose

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)


@dataclass
class BuildRunCli:
    """CLI surface for `codeintel build run`."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to build (e.g., function_metrics, call_graph).",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Build all targets in a module (ingestion, graphs, analytics).",
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Build all targets across all modules.",
            negative=(),
        ),
    ] = False
    dry_run: Annotated[
        bool,
        Parameter(
            name=["--dry-run", "-n"],
            help="Show build plan without executing.",
            negative=(),
        ),
    ] = False
    force: Annotated[
        list[str] | None,
        Parameter(
            name=["--force", "-f"],
            help="Force recompute of specific targets (repeatable).",
        ),
    ] = None
    project_root: ProjectRoot = None
    verbose: Verbose = 0
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False


@build_app.command(name="run")
def build_run(
    cfg: Annotated[BuildRunCli, Parameter(name="*")] | None = None,
) -> None:
    """Build targets with automatic dependency resolution.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    cfg = cfg or BuildRunCli()
    output_format = cfg.output_format
    if cfg.json:
        output_format = OutputFormat.JSON

    options = BuildRunOptions(
        targets=cfg.targets,
        module=cfg.module,
        target_scope=TargetScope.ALL if cfg.all_targets else TargetScope.REQUESTED,
        run_mode=RunMode.DRY_RUN if cfg.dry_run else RunMode.EXECUTE,
        force=cfg.force,
    )
    ctx_opts = BuildRunContext(
        runtime_options=RuntimeCliOptions(project_root=cfg.project_root),
        verbose=cfg.verbose,
        output_format=output_format,
    )

    try:
        build_run_handler(options, ctx_opts)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


@dataclass
class BuildStatusCli:
    """CLI surface for `codeintel build status`."""

    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Filter status to a specific module (ingestion, graphs, analytics).",
        ),
    ] = None
    project_root: ProjectRoot = None
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False
    verbose: Verbose = 0


@build_app.command(name="status")
def build_status(
    cfg: Annotated[BuildStatusCli, Parameter(name="*")] | None = None,
) -> None:
    """Show current state of build targets.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    cfg = cfg or BuildStatusCli()
    output_format = cfg.output_format
    if cfg.json:
        output_format = OutputFormat.JSON

    options = BuildStatusOptions(
        module=cfg.module,
        runtime_options=RuntimeCliOptions(project_root=cfg.project_root),
        output_format=output_format,
        verbose=cfg.verbose,
    )
    try:
        build_status_handler(options)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


@dataclass
class BuildHistoryCli:
    """CLI surface for `codeintel build history`."""

    run_id: Annotated[
        str | None,
        Parameter(
            name=["--run-id", "-i"],
            help="Specific run ID to show details for (prefix match supported).",
        ),
    ] = None
    limit: Annotated[
        int,
        Parameter(
            name=["--limit", "-n"],
            help="Number of recent runs to show.",
        ),
    ] = 10
    project_root: ProjectRoot = None
    output_format: OutputFmt = OutputFormat.TEXT
    json: JsonFlag = False
    verbose: Verbose = 0


@build_app.command(name="history")
def build_history(
    cfg: Annotated[BuildHistoryCli, Parameter(name="*")] | None = None,
) -> None:
    """Show build run history and details.

    Raises
    ------
    SystemExit
        When the underlying handler triggers a CLI exit.
    """
    cfg = cfg or BuildHistoryCli()
    output_format = cfg.output_format
    if cfg.json:
        output_format = OutputFormat.JSON

    options = BuildHistoryOptions(
        run_id=cfg.run_id,
        limit=cfg.limit,
        runtime_options=RuntimeCliOptions(project_root=cfg.project_root),
        output_format=output_format,
        verbose=cfg.verbose,
    )

    try:
        build_history_handler(options)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


__all__ = ["build_app"]
