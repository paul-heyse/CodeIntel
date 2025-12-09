"""Cyclopts wiring for subsystem exploration commands."""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands.subsystem import (
    SubsystemCoverageOptions,
    SubsystemIdOptions,
    SubsystemListOptions,
    SubsystemMembershipOptions,
    SubsystemProfilesOptions,
    SubsystemRuntime,
    subsystem_coverage_handler,
    subsystem_list_handler,
    subsystem_module_memberships_handler,
    subsystem_profiles_handler,
    subsystem_show_handler,
)
from codeintel.cli.cyclopts_common import RuntimeCLI, runtime_cli_to_options

subsystem_app = App(
    name="subsystem",
    help="Subsystem exploration commands.",
)


def _runtime(cli: RuntimeCLI) -> SubsystemRuntime:
    return SubsystemRuntime(runtime_options=runtime_cli_to_options(cli), verbose=cli.verbose)


@subsystem_app.command(name="list")
def list_subsystems(
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
    role: Annotated[
        str | None,
        Parameter(
            name="--role",
            help="Filter subsystems by role tag.",
        ),
    ] = None,
    query: Annotated[
        str | None,
        Parameter(
            name="--q",
            help="Search substring on name/description.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of subsystems returned.",
        ),
    ] = None,
) -> None:
    """List inferred subsystems with role/risk metadata."""
    cfg = runtime or RuntimeCLI()
    options = SubsystemListOptions(
        runtime=_runtime(cfg),
        role=role,
        query=query,
        limit=limit,
    )
    subsystem_list_handler(options)


@subsystem_app.command(name="show")
def show_subsystem(
    subsystem_id: Annotated[
        str,
        Parameter(
            name=None,
            help="Subsystem identifier.",
        ),
    ],
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
) -> None:
    """Show subsystem detail and modules."""
    cfg = runtime or RuntimeCLI()
    options = SubsystemIdOptions(
        runtime=_runtime(cfg),
        subsystem_id=subsystem_id,
    )
    subsystem_show_handler(options)


@subsystem_app.command(name="profiles")
def list_profiles(
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of profiles returned.",
        ),
    ] = None,
) -> None:
    """List subsystem profiles from docs.v_subsystem_profile."""
    cfg = runtime or RuntimeCLI()
    options = SubsystemProfilesOptions(runtime=_runtime(cfg), limit=limit)
    subsystem_profiles_handler(options)


@subsystem_app.command(name="coverage")
def list_coverage(
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of coverage rows returned.",
        ),
    ] = None,
) -> None:
    """List subsystem coverage rollups from docs.v_subsystem_coverage."""
    cfg = runtime or RuntimeCLI()
    options = SubsystemCoverageOptions(runtime=_runtime(cfg), limit=limit)
    subsystem_coverage_handler(options)


@subsystem_app.command(name="module-memberships")
def module_memberships(
    module: Annotated[
        str,
        Parameter(
            name=None,
            help="Module name (e.g., pkg.mod).",
        ),
    ],
    runtime: Annotated[RuntimeCLI, Parameter(name="*")] | None = None,
) -> None:
    """List subsystem memberships for a module."""
    cfg = runtime or RuntimeCLI()
    options = SubsystemMembershipOptions(runtime=_runtime(cfg), module=module)
    subsystem_module_memberships_handler(options)


__all__ = ["subsystem_app"]
