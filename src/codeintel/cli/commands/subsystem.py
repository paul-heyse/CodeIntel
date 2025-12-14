"""Subsystem exploration commands.

Note: Subsystem commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
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


_SUBSYSTEM_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("subsystem.list", handler=subsystem_list_handler, config=_SUBSYSTEM_CONFIG)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("subsystem.show", handler=subsystem_show_handler, config=_SUBSYSTEM_CONFIG)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("subsystem.profiles", handler=subsystem_profiles_handler, config=_SUBSYSTEM_CONFIG)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command("subsystem.coverage", handler=subsystem_coverage_handler, config=_SUBSYSTEM_CONFIG)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


@cli_command(
    "subsystem.module_memberships",
    handler=subsystem_module_memberships_handler,
    config=_SUBSYSTEM_CONFIG,
)
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
    flags: SharedFlags = field(default_factory=SharedFlags, metadata=SHARED_FLAGS_METADATA)


__all__ = ["subsystem_app"]
