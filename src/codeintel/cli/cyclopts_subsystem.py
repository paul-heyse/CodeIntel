"""Cyclopts wiring for subsystem exploration commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.execution.adapter import CycloptsAdapter
from codeintel.cli.subsystem_handlers import (
    subsystem_coverage_ctx,
    subsystem_list_ctx,
    subsystem_module_memberships_ctx,
    subsystem_profiles_ctx,
    subsystem_show_ctx,
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
        CycloptsAdapter("subsystem.list", subsystem_list_ctx)(self)


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
        CycloptsAdapter("subsystem.show", subsystem_show_ctx)(self)


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
        CycloptsAdapter("subsystem.profiles", subsystem_profiles_ctx)(self)


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
        CycloptsAdapter("subsystem.coverage", subsystem_coverage_ctx)(self)


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
        CycloptsAdapter("subsystem.module_memberships", subsystem_module_memberships_ctx)(self)


__all__ = ["subsystem_app"]
