"""Cyclopts wiring for subsystem exploration commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import RuntimeCliOptions
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
from codeintel.cli.cyclopts_common import ProjectRoot, Verbose

subsystem_app = App(
    name="subsystem",
    help="Subsystem exploration commands.",
)


@dataclass
class SubsystemRuntimeCli:
    """Runtime selection for subsystem commands."""

    project_root: ProjectRoot = None
    repo: Annotated[
        str | None,
        Parameter(
            name="--repo",
            help="Repository slug (e.g., 'org/repo'). Uses project config if omitted.",
        ),
    ] = None
    commit: Annotated[
        str | None,
        Parameter(
            name="--commit",
            help="Commit SHA. Uses project config if omitted.",
        ),
    ] = None
    db_path: Annotated[
        Path | None,
        Parameter(
            name="--db-path",
            help="Path to DuckDB database. Uses project config if omitted.",
        ),
    ] = None
    build_dir: Annotated[
        Path | None,
        Parameter(
            name="--build-dir",
            help="Build directory (default: build/).",
        ),
    ] = None
    repo_root: Annotated[
        Path | None,
        Parameter(
            name="--repo-root",
            help="Path to repository root (default: current directory).",
        ),
    ] = None
    verbose: Verbose = 0


def _runtime(cli: SubsystemRuntimeCli) -> SubsystemRuntime:
    return SubsystemRuntime(
        runtime_options=RuntimeCliOptions(
            project_root=cli.project_root,
            repo=cli.repo,
            commit=cli.commit,
            db_path=cli.db_path,
            build_dir=cli.build_dir,
            repo_root=cli.repo_root,
        ),
        verbose=cli.verbose,
    )


@subsystem_app.command(name="list")
def list_subsystems(
    runtime: Annotated[SubsystemRuntimeCli, Parameter(name="*")] | None = None,
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
    cfg = runtime or SubsystemRuntimeCli()
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
    runtime: Annotated[SubsystemRuntimeCli, Parameter(name="*")] | None = None,
) -> None:
    """Show subsystem detail and modules."""
    cfg = runtime or SubsystemRuntimeCli()
    options = SubsystemIdOptions(
        runtime=_runtime(cfg),
        subsystem_id=subsystem_id,
    )
    subsystem_show_handler(options)


@subsystem_app.command(name="profiles")
def list_profiles(
    runtime: Annotated[SubsystemRuntimeCli, Parameter(name="*")] | None = None,
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of profiles returned.",
        ),
    ] = None,
) -> None:
    """List subsystem profiles from docs.v_subsystem_profile."""
    cfg = runtime or SubsystemRuntimeCli()
    options = SubsystemProfilesOptions(runtime=_runtime(cfg), limit=limit)
    subsystem_profiles_handler(options)


@subsystem_app.command(name="coverage")
def list_coverage(
    runtime: Annotated[SubsystemRuntimeCli, Parameter(name="*")] | None = None,
    limit: Annotated[
        int | None,
        Parameter(
            name="--limit",
            help="Limit the number of coverage rows returned.",
        ),
    ] = None,
) -> None:
    """List subsystem coverage rollups from docs.v_subsystem_coverage."""
    cfg = runtime or SubsystemRuntimeCli()
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
    runtime: Annotated[SubsystemRuntimeCli, Parameter(name="*")] | None = None,
) -> None:
    """List subsystem memberships for a module."""
    cfg = runtime or SubsystemRuntimeCli()
    options = SubsystemMembershipOptions(runtime=_runtime(cfg), module=module)
    subsystem_module_memberships_handler(options)


__all__ = ["subsystem_app"]
