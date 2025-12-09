"""Cyclopts wiring for the build command group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.build_handlers import (
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
from codeintel.cli.cli_errors import ValidationError, run_handler
from codeintel.cli.common_handlers import OutputFormat
from codeintel.cli.cyclopts_common import (
    OutputFormatCLI,
    OutputParam,
    RuntimeCLI,
    RuntimeParam,
    resolve_output_format,
    runtime_cli_to_options,
)

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)

MODULE_CHOICES = ("ingestion", "graphs", "analytics")


def _validate_build_run_selection(
    targets: list[str] | None, module: str | None, *, all_targets: bool
) -> None:
    """Enforce exactly one of targets/module/all_targets.

    Raises
    ------
    ValidationError
        If zero or multiple selection mechanisms are provided.
    """
    provided = [
        bool(targets),
        module is not None,
        all_targets,
    ]
    if sum(provided) != 1:
        message = "Provide exactly one of targets, --module, or --all."
        raise ValidationError(message)


@build_app.command(name="run")
@dataclass
class BuildRunCli:
    """Build targets with automatic dependency resolution.

    Raises
    ------
    ValidationError
        If selection flags are invalid or module is unknown.
    """

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
            show_choices=True,
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
    runtime: RuntimeParam = field(default_factory=RuntimeCLI)
    output: OutputParam = field(default_factory=OutputFormatCLI)

    def __call__(self) -> None:
        if self.module is not None and self.module not in MODULE_CHOICES:
            valid = ", ".join(MODULE_CHOICES)
            message = f"Unknown module: {self.module}. Valid: {valid}"
            raise ValidationError(message)
        _validate_build_run_selection(self.targets, self.module, all_targets=self.all_targets)
        runtime_opts = runtime_cli_to_options(self.runtime)
        output_format = resolve_output_format(
            json_flag=self.output.json,
            explicit=self.output.output_format,
            default=OutputFormat.TEXT,
        )
        options = BuildRunOptions(
            targets=self.targets,
            module=self.module,
            target_scope=TargetScope.ALL if self.all_targets else TargetScope.REQUESTED,
            run_mode=RunMode.DRY_RUN if self.dry_run else RunMode.EXECUTE,
            force=self.force,
        )
        ctx_opts = BuildRunContext(
            runtime_options=runtime_opts,
            verbose=self.runtime.verbose,
            output_format=output_format,
        )
        run_handler(build_run_handler, options, ctx_opts)


@build_app.command(name="status")
@dataclass
class BuildStatusCli:
    """Show current state of build targets."""

    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Filter status to a specific module (ingestion, graphs, analytics).",
            show_choices=True,
        ),
    ] = None
    runtime: RuntimeParam = field(default_factory=RuntimeCLI)
    output: OutputParam = field(default_factory=OutputFormatCLI)

    def __call__(self) -> None:
        if self.module is not None and self.module not in MODULE_CHOICES:
            valid = ", ".join(MODULE_CHOICES)
            message = f"Unknown module: {self.module}. Valid: {valid}"
            raise ValidationError(message)
        runtime_opts = runtime_cli_to_options(self.runtime)
        output_format = resolve_output_format(
            json_flag=self.output.json,
            explicit=self.output.output_format,
            default=OutputFormat.TEXT,
        )
        options = BuildStatusOptions(
            module=self.module,
            runtime_options=runtime_opts,
            output_format=output_format,
            verbose=self.runtime.verbose,
        )
        run_handler(build_status_handler, options)


@build_app.command(name="history")
@dataclass
class BuildHistoryCli:
    """Show build run history and details."""

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
    runtime: RuntimeParam = field(default_factory=RuntimeCLI)
    output: OutputParam = field(default_factory=OutputFormatCLI)

    def __call__(self) -> None:
        runtime_opts = runtime_cli_to_options(self.runtime)
        output_format = resolve_output_format(
            json_flag=self.output.json,
            explicit=self.output.output_format,
            default=OutputFormat.TEXT,
        )
        options = BuildHistoryOptions(
            run_id=self.run_id,
            limit=self.limit,
            runtime_options=runtime_opts,
            output_format=output_format,
            verbose=self.runtime.verbose,
        )
        run_handler(build_history_handler, options)


__all__ = ["build_app"]
