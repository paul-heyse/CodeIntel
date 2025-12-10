"""Cyclopts wiring for the build command group.

This module wires Cyclopts command classes to unified handlers via command_context.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.commands.context import command_context
from codeintel.cli.handlers.build import (
    build_history_handler,
    build_run_handler,
    build_status_handler,
)
from codeintel.cli.rendering.types import OutputFormat

build_app = App(
    name="build",
    help="Build system commands for minimal-work target computation.",
)


@build_app.command(name="run")
@dataclass
class BuildRunCommand:
    """Build targets with automatic dependency resolution."""

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
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the build run command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "targets": self.targets,
            "module": self.module,
            "all_targets": self.all_targets,
            "dry_run": self.dry_run,
            "force": self.force,
        }

        with command_context(
            "build.run",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = build_run_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@build_app.command(name="status")
@dataclass
class BuildStatusCommand:
    """Show current state of build targets."""

    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Filter status to a specific module (ingestion, graphs, analytics).",
            show_choices=True,
        ),
    ] = None
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the build status command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "module": self.module,
        }

        with command_context(
            "build.status",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = build_status_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@build_app.command(name="history")
@dataclass
class BuildHistoryCommand:
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
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    output_format: Annotated[
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format (text or json).",
        ),
    ] = OutputFormat.TEXT
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the build history command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "run_id": self.run_id,
            "limit": self.limit,
        }

        with command_context(
            "build.history",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = build_history_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["build_app"]
