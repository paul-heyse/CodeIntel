"""Subsystem exploration commands for the CodeIntel CLI.

This module provides Typer commands for exploring and querying
inferred subsystems with role and risk metadata.

Commands
--------
- **list**: List inferred subsystems with role/risk metadata
- **show**: Show subsystem detail and modules
- **profiles**: List subsystem profiles
- **coverage**: List subsystem coverage rollups
- **modules**: List subsystem memberships for a module
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from codeintel.cli.commands._common import (
    BuildDirOpt,
    CommitOpt,
    DbPathOpt,
    LimitOpt,
    ProjectRootOpt,
    RepoOpt,
    RepoRootOpt,
    RuntimeCliOptions,
    VerboseOpt,
    build_graph_runtime,
    build_runtime_or_exit,
    open_gateway_from_config,
    setup_logging,
)
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import (
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemProfileResponse,
    SubsystemProfileRow,
)

LOG = logging.getLogger(__name__)

subsystem_app = typer.Typer(
    name="subsystem",
    help="Subsystem exploration commands.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

RoleOpt = Annotated[
    str | None,
    typer.Option("--role", help="Filter subsystems by role tag"),
]

QueryOpt = Annotated[
    str | None,
    typer.Option("--q", help="Search substring on name/description"),
]

SubsystemIdArg = Annotated[
    str,
    typer.Argument(help="Subsystem identifier"),
]

ModuleArg = Annotated[
    str,
    typer.Argument(help="Module name (e.g., pkg.mod)"),
]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _build_backend(
    project_root: Path | None,
    repo: RepoOpt,
    commit: CommitOpt,
    db_path: DbPathOpt,
    build_dir: BuildDirOpt,
    repo_root: RepoRootOpt,
    verbose: int = VerboseOpt,
) -> DuckDBBackend:
    """Build a DuckDBBackend from CLI options.

    Parameters
    ----------
    project_root
        Project root path.
    repo
        Repository slug.
    commit
        Commit SHA.
    db_path
        Database path.
    build_dir
        Build directory.
    repo_root
        Repository root.
    verbose
        Verbosity level.

    Returns
    -------
    DuckDBBackend
        Constructed backend.
    """
    setup_logging(verbose)

    runtime_options = RuntimeCliOptions(
        project_root=project_root,
        repo=repo,
        commit=commit,
        db_path=db_path,
        build_dir=build_dir,
        repo_root=repo_root,
    )
    runtime = build_runtime_or_exit(runtime_options)

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    graph_runtime = build_graph_runtime(runtime.cfg, gateway)

    resource = build_backend_resource(
        runtime.serving,
        gateway=gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    # The backend is guaranteed to be DuckDBBackend for local_db mode
    backend = resource.backend
    if not isinstance(backend, DuckDBBackend):
        msg = "Expected DuckDBBackend for local_db mode"
        raise TypeError(msg)
    return backend


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@subsystem_app.command("list")
def subsystem_list(
    project_root: Path | None = ProjectRootOpt,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    role: RoleOpt = None,
    q: QueryOpt = None,
    limit: LimitOpt = None,
    verbose: int = VerboseOpt,
) -> None:
    """List inferred subsystems with role/risk metadata.

    Shows subsystems from cached docs views with optional filtering
    by role and search query.

    Examples
    --------
    .. code-block:: bash

        # List all subsystems
        codeintel subsystem list

        # Filter by role
        codeintel subsystem list --role core

        # Search by name
        codeintel subsystem list --q analytics
    """
    backend = _build_backend(project_root, repo, commit, db_path, build_dir, repo_root, verbose)
    response = backend.list_subsystems(
        limit=limit,
        role=role,
        q=q,
    )
    payload = {
        "subsystems": [row.model_dump() for row in response.subsystems],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


@subsystem_app.command("show")
def subsystem_show(
    subsystem_id: SubsystemIdArg,
    project_root: Path | None = ProjectRootOpt,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    verbose: int = VerboseOpt,
) -> None:
    """Show subsystem detail and modules.

    Displays detailed information about a specific subsystem including
    its member modules.

    Examples
    --------
    .. code-block:: bash

        # Show subsystem details
        codeintel subsystem show analytics_core
    """
    backend = _build_backend(project_root, repo, commit, db_path, build_dir, repo_root, verbose)
    response = backend.get_subsystem_modules(subsystem_id=subsystem_id)

    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", subsystem_id)
        typer.secho(f"Subsystem not found: {subsystem_id}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    payload = {
        "subsystem": response.subsystem.model_dump(),
        "modules": [row.model_dump() for row in response.modules],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


@subsystem_app.command("profiles")
def subsystem_profiles(
    project_root: Path | None = ProjectRootOpt,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    limit: LimitOpt = None,
    verbose: int = VerboseOpt,
) -> None:
    """List subsystem profiles from docs.v_subsystem_profile.

    Shows subsystem profile data from read-only docs views.

    Examples
    --------
    .. code-block:: bash

        # List profiles
        codeintel subsystem profiles

        # Limit results
        codeintel subsystem profiles --limit 10
    """
    backend = _build_backend(project_root, repo, commit, db_path, build_dir, repo_root, verbose)
    response = backend.service.list_subsystem_profiles(limit=limit)
    profile_response = (
        response
        if isinstance(response, SubsystemProfileResponse)
        else SubsystemProfileResponse.from_domain(response)
    )
    profiles = [
        row if isinstance(row, SubsystemProfileRow) else SubsystemProfileRow.model_validate(row)
        for row in profile_response.profiles
    ]
    payload = {
        "profiles": [row.model_dump() for row in profiles],
        "meta": profile_response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


@subsystem_app.command("coverage")
def subsystem_coverage(
    project_root: Path | None = ProjectRootOpt,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    limit: LimitOpt = None,
    verbose: int = VerboseOpt,
) -> None:
    """List subsystem coverage rollups from docs.v_subsystem_coverage.

    Shows subsystem coverage data from read-only docs views.

    Examples
    --------
    .. code-block:: bash

        # List coverage
        codeintel subsystem coverage

        # Limit results
        codeintel subsystem coverage --limit 10
    """
    backend = _build_backend(project_root, repo, commit, db_path, build_dir, repo_root, verbose)
    response = backend.service.list_subsystem_coverage(limit=limit)
    coverage_response = (
        response
        if isinstance(response, SubsystemCoverageResponse)
        else SubsystemCoverageResponse.from_domain(response)
    )
    coverage_rows = [
        row if isinstance(row, SubsystemCoverageRow) else SubsystemCoverageRow.model_validate(row)
        for row in coverage_response.coverage
    ]
    payload = {
        "coverage": [row.model_dump() for row in coverage_rows],
        "meta": coverage_response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


@subsystem_app.command("module-memberships")
def subsystem_module_memberships(
    module: ModuleArg,
    project_root: Path | None = ProjectRootOpt,
    repo: RepoOpt = None,
    commit: CommitOpt = None,
    db_path: DbPathOpt = None,
    build_dir: BuildDirOpt = None,
    repo_root: RepoRootOpt = None,
    verbose: int = VerboseOpt,
) -> None:
    """List subsystem memberships for a module.

    Shows which subsystems a given module belongs to.

    Examples
    --------
    .. code-block:: bash

        # Get memberships for a module
        codeintel subsystem module-memberships pkg.mod
    """
    backend = _build_backend(project_root, repo, commit, db_path, build_dir, repo_root, verbose)
    response = backend.get_module_subsystems(module=module)
    payload = {
        "found": response.found,
        "memberships": [row.model_dump() for row in response.memberships],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


__all__ = ["subsystem_app"]
