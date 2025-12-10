"""Cyclopts wiring for subsystem exploration commands.

This module wires Cyclopts command classes to unified EnhancedHandlerContext handlers.
Commands use the command_context() helper for standardized infrastructure:

- Configuration loading via ConfigService
- Runtime resolution
- Logging setup based on verbosity
- Unified rendering via UnifiedRenderer
- Automatic resource cleanup
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.cli_types import OutputFormat
from codeintel.cli.command_context import command_context
from codeintel.cli.cyclopts_common import OutputFormatCLI, RuntimeCLI
from codeintel.cli.handlers.subsystem import (
    subsystem_coverage_handler,
    subsystem_list_handler,
    subsystem_module_memberships_handler,
    subsystem_profiles_handler,
    subsystem_show_handler,
)

subsystem_app = App(
    name="subsystem",
    help="Subsystem exploration commands.",
)


@subsystem_app.command(name="list")
@dataclass
class SubsystemListCommand:
    """List inferred subsystems with role/risk metadata."""

    role: Annotated[
        str | None,
        Parameter(
            name="--role",
            help="Filter subsystems by role tag.",
        ),
    ] = None
    query: Annotated[
        str | None,
        Parameter(
            name="--q",
            help="Search substring on name/description.",
        ),
    ] = None
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of subsystems returned.",
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
        """Execute the subsystem list command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {
            "role": self.role,
            "query": self.query,
            "limit": self.limit,
        }

        with command_context(
            "subsystem.list",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = subsystem_list_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@subsystem_app.command(name="show")
@dataclass
class SubsystemShowCommand:
    """Show subsystem detail and modules."""

    subsystem_id: Annotated[
        str,
        Parameter(
            name=None,
            help="Subsystem identifier.",
        ),
    ] = ""
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
        """Execute the subsystem show command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"subsystem_id": self.subsystem_id}

        with command_context(
            "subsystem.show",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = subsystem_show_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@subsystem_app.command(name="profiles")
@dataclass
class SubsystemProfilesCommand:
    """List subsystem profiles from docs.v_subsystem_profile."""

    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of profiles returned.",
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
        """Execute the subsystem profiles command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"limit": self.limit}

        with command_context(
            "subsystem.profiles",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = subsystem_profiles_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@subsystem_app.command(name="coverage")
@dataclass
class SubsystemCoverageCommand:
    """List subsystem coverage rollups from docs.v_subsystem_coverage."""

    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of coverage rows returned.",
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
        """Execute the subsystem coverage command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"limit": self.limit}

        with command_context(
            "subsystem.coverage",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = subsystem_coverage_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


@subsystem_app.command(name="module-memberships")
@dataclass
class SubsystemMembershipCommand:
    """List subsystem memberships for a module."""

    module: Annotated[
        str,
        Parameter(
            name=None,
            help="Module name (e.g., pkg.mod).",
        ),
    ] = ""
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
        """Execute the subsystem module-memberships command."""
        runtime_cli = RuntimeCLI(
            project_root=self.project_root,
            verbose=self.verbose,
        )
        output_cli = OutputFormatCLI(output_format=self.output_format)

        params: dict[str, object] = {"module": self.module}

        with command_context(
            "subsystem.module_memberships",
            runtime_cli,
            output_cli,
            params=params,
        ) as (ctx, renderer):
            result = subsystem_module_memberships_handler(ctx)
            exit_code = renderer.render_result(result)
            if exit_code != 0:
                sys.exit(exit_code)


__all__ = ["subsystem_app"]
