"""Cyclopts wiring for the build command group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import ValidationError, invoke_with_typer_translation
from codeintel.cli.commands._common import OutputFormat
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
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    RuntimeCLI,
    resolve_output_format,
    runtime_cli_to_options,
)

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)

MODULE_CHOICES = ("ingestion", "graphs", "analytics")


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)


@build_app.command(name="run")
def build_run(
    cfg: Annotated[BuildRunCli, Parameter(name="*")] | None = None,
) -> None:
    """Build targets with automatic dependency resolution.

    Raises
    ------
    ValidationError
        If an unknown module value is provided.
    """
    cfg = cfg or BuildRunCli()
    if cfg.module is not None and cfg.module not in MODULE_CHOICES:
        valid = ", ".join(MODULE_CHOICES)
        message = f"Unknown module: {cfg.module}. Valid: {valid}"
        raise ValidationError(message)
    runtime_opts = runtime_cli_to_options(cfg.runtime)
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )

    options = BuildRunOptions(
        targets=cfg.targets,
        module=cfg.module,
        target_scope=TargetScope.ALL if cfg.all_targets else TargetScope.REQUESTED,
        run_mode=RunMode.DRY_RUN if cfg.dry_run else RunMode.EXECUTE,
        force=cfg.force,
    )
    ctx_opts = BuildRunContext(
        runtime_options=runtime_opts,
        verbose=cfg.runtime.verbose,
        output_format=output_format,
    )
    invoke_with_typer_translation(build_run_handler, options, ctx_opts)


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)


@build_app.command(name="status")
def build_status(
    cfg: Annotated[BuildStatusCli, Parameter(name="*")] | None = None,
) -> None:
    """Show current state of build targets.

    Raises
    ------
    ValidationError
        If an unknown module value is provided.
    """
    cfg = cfg or BuildStatusCli()
    if cfg.module is not None and cfg.module not in MODULE_CHOICES:
        valid = ", ".join(MODULE_CHOICES)
        message = f"Unknown module: {cfg.module}. Valid: {valid}"
        raise ValidationError(message)
    runtime_opts = runtime_cli_to_options(cfg.runtime)
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )

    options = BuildStatusOptions(
        module=cfg.module,
        runtime_options=runtime_opts,
        output_format=output_format,
        verbose=cfg.runtime.verbose,
    )
    invoke_with_typer_translation(build_status_handler, options)


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
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)
    output: Annotated[OutputFormatCLI, Parameter(name="*")] = field(default_factory=OutputFormatCLI)


@build_app.command(name="history")
def build_history(
    cfg: Annotated[BuildHistoryCli, Parameter(name="*")] | None = None,
) -> None:
    """Show build run history and details."""
    cfg = cfg or BuildHistoryCli()
    runtime_opts = runtime_cli_to_options(cfg.runtime)
    output_format = resolve_output_format(
        json_flag=cfg.output.json,
        explicit=cfg.output.output_format,
        default=OutputFormat.TEXT,
    )

    options = BuildHistoryOptions(
        run_id=cfg.run_id,
        limit=cfg.limit,
        runtime_options=runtime_opts,
        output_format=output_format,
        verbose=cfg.runtime.verbose,
    )
    invoke_with_typer_translation(build_history_handler, options)


__all__ = ["build_app"]
