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
from dataclasses import dataclass

import typer

from codeintel.cli.commands._common import (
    RuntimeCliOptions,
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


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemRuntime:
    """Runtime inputs shared by subsystem commands."""

    runtime_options: RuntimeCliOptions
    verbose: int = 0


@dataclass(frozen=True)
class SubsystemListOptions:
    """Options for listing subsystems."""

    runtime: SubsystemRuntime
    role: str | None = None
    query: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class SubsystemIdOptions:
    """Options for subsystem detail commands."""

    runtime: SubsystemRuntime
    subsystem_id: str


@dataclass(frozen=True)
class SubsystemProfilesOptions:
    """Options for subsystem profile listing."""

    runtime: SubsystemRuntime
    limit: int | None = None


@dataclass(frozen=True)
class SubsystemCoverageOptions:
    """Options for subsystem coverage listing."""

    runtime: SubsystemRuntime
    limit: int | None = None


@dataclass(frozen=True)
class SubsystemMembershipOptions:
    """Options for module membership queries."""

    runtime: SubsystemRuntime
    module: str


def _build_backend(runtime: SubsystemRuntime) -> DuckDBBackend:
    """Build a DuckDBBackend from runtime options.

    Returns
    -------
    DuckDBBackend
        Constructed backend instance.

    Raises
    ------
    TypeError
        If the resolved backend is not a DuckDBBackend.
    """
    setup_logging(runtime.verbose)

    runtime_cfg = build_runtime_or_exit(runtime.runtime_options)
    gateway = open_gateway_from_config(runtime_cfg.cfg, read_only=True)
    graph_runtime = build_graph_runtime(runtime_cfg.cfg, gateway)

    resource = build_backend_resource(
        runtime_cfg.serving,
        gateway=gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    backend = resource.backend
    if not isinstance(backend, DuckDBBackend):
        msg = "Expected DuckDBBackend for local_db mode"
        raise TypeError(msg)
    return backend


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def subsystem_list_handler(options: SubsystemListOptions) -> None:
    """List inferred subsystems with role/risk metadata."""
    backend = _build_backend(options.runtime)
    response = backend.list_subsystems(
        limit=options.limit,
        role=options.role,
        q=options.query,
    )
    payload = {
        "subsystems": [row.model_dump() for row in response.subsystems],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


def subsystem_show_handler(options: SubsystemIdOptions) -> None:
    """Show subsystem detail and modules.

    Raises
    ------
    typer.Exit
        If the subsystem cannot be found.
    """
    backend = _build_backend(options.runtime)
    response = backend.get_subsystem_modules(subsystem_id=options.subsystem_id)

    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", options.subsystem_id)
        typer.secho(
            f"Subsystem not found: {options.subsystem_id}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    payload = {
        "subsystem": response.subsystem.model_dump(),
        "modules": [row.model_dump() for row in response.modules],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


def subsystem_profiles_handler(options: SubsystemProfilesOptions) -> None:
    """List subsystem profiles from docs.v_subsystem_profile."""
    backend = _build_backend(options.runtime)
    response = backend.service.list_subsystem_profiles(limit=options.limit)
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


def subsystem_coverage_handler(options: SubsystemCoverageOptions) -> None:
    """List subsystem coverage rollups from docs.v_subsystem_coverage."""
    backend = _build_backend(options.runtime)
    response = backend.service.list_subsystem_coverage(limit=options.limit)
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


def subsystem_module_memberships_handler(options: SubsystemMembershipOptions) -> None:
    """List subsystem memberships for a module."""
    backend = _build_backend(options.runtime)
    response = backend.get_module_subsystems(module=options.module)
    payload = {
        "found": response.found,
        "memberships": [row.model_dump() for row in response.memberships],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


__all__ = [
    "SubsystemCoverageOptions",
    "SubsystemIdOptions",
    "SubsystemListOptions",
    "SubsystemMembershipOptions",
    "SubsystemProfilesOptions",
    "SubsystemRuntime",
    "subsystem_coverage_handler",
    "subsystem_list_handler",
    "subsystem_module_memberships_handler",
    "subsystem_profiles_handler",
    "subsystem_show_handler",
]
