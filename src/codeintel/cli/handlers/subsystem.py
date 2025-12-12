"""Subsystem exploration handlers following the unified handler pattern.

This module provides handlers for subsystem commands using the
CommandContext pattern for consistent resource management
and output rendering.

All handlers in this module:

1. Accept CommandContext as their only argument
2. Return CliResult[T]
3. Never write to stdout/stderr directly
4. Never call sys.exit()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.analytics.runtime import GraphRuntimeOptions, build_graph_runtime
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_subsystem_not_found
from codeintel.config.primitives import SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import (
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemProfileResponse,
    SubsystemProfileRow,
)

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubsystemListResult:
    """Result from subsystem list operation.

    Parameters
    ----------
    subsystems
        List of subsystem data.
    meta
        Response metadata.
    """

    subsystems: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "subsystems": self.subsystems,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemShowResult:
    """Result from subsystem show operation.

    Parameters
    ----------
    subsystem
        Subsystem detail data.
    modules
        List of module data in the subsystem.
    meta
        Response metadata.
    """

    subsystem: dict[str, Any]
    modules: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "subsystem": self.subsystem,
            "modules": self.modules,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemProfilesResult:
    """Result from subsystem profiles operation.

    Parameters
    ----------
    profiles
        List of profile data.
    meta
        Response metadata.
    """

    profiles: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "profiles": self.profiles,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemCoverageResult:
    """Result from subsystem coverage operation.

    Parameters
    ----------
    coverage
        List of coverage data.
    meta
        Response metadata.
    """

    coverage: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "coverage": self.coverage,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class SubsystemMembershipResult:
    """Result from subsystem membership operation.

    Parameters
    ----------
    found
        Whether the module was found.
    memberships
        List of membership data.
    meta
        Response metadata.
    """

    found: bool
    memberships: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "found": self.found,
            "memberships": self.memberships,
            "meta": self.meta,
        }


def _build_backend(ctx: CommandContext) -> DuckDBBackend:
    """Build a DuckDBBackend from context.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    DuckDBBackend
        Constructed backend instance.

    Raises
    ------
    TypeError
        If resolved backend is not DuckDBBackend.
    """
    gateway_backend = getattr(ctx.gateway, "backend", None)
    if gateway_backend is not None:
        if isinstance(gateway_backend, DuckDBBackend):
            return gateway_backend
        message = f"Expected DuckDBBackend, got {type(gateway_backend).__name__}"
        raise TypeError(message)

    backend_override = getattr(ctx, "_backend_override", None) or ctx.params.raw.get(
        "_backend_override"
    )
    if backend_override is not None:
        if isinstance(backend_override, DuckDBBackend):
            return backend_override
        message = f"Expected DuckDBBackend override, got {type(backend_override).__name__}"
        raise TypeError(message)

    gateway_config = getattr(ctx.gateway, "config", None)
    repo_root = ctx.params.get_path("repo_root") or Path.cwd()
    repo = getattr(gateway_config, "repo", None) or ctx.params.get_str("repo")
    if not isinstance(repo, str) or not repo:
        repo = repo_root.name
    commit = getattr(gateway_config, "commit", None) or ctx.params.get_str("commit")
    if not isinstance(commit, str) or not commit:
        commit = "HEAD"

    snapshot = (
        ctx.runtime.snapshot
        if ctx.has_runtime
        else SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    )

    graph_runtime = build_graph_runtime(
        gateway=ctx.gateway,
        options=GraphRuntimeOptions(snapshot=snapshot),
    )

    serving_cfg = (
        ctx.runtime.serving
        if ctx.has_runtime
        else ServingConfig(
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            db_path=getattr(gateway_config, "db_path", None),
        )
    )

    resource = build_backend_resource(
        serving_cfg,
        gateway=ctx.gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    backend = resource.backend
    if not isinstance(backend, DuckDBBackend):
        msg = "Expected DuckDBBackend for local_db mode"
        raise TypeError(msg)
    return backend


def subsystem_list_handler(ctx: CommandContext) -> CliResult[SubsystemListResult]:
    """List inferred subsystems with role/risk metadata.

    Parameters
    ----------
    ctx
        Command context with optional params:

        - role: Optional role filter
        - query: Optional search query
        - limit: Optional result limit

    Returns
    -------
    CliResult[SubsystemListResult]
        Result with subsystem list.

    Raises
    ------
    TypeError
        If a backend override is provided that is not a DuckDBBackend.
    """
    backend_override = getattr(ctx, "_backend_override", None) or ctx.params.raw.get(
        "_backend_override"
    )
    if backend_override is not None:
        if not isinstance(backend_override, DuckDBBackend):
            message = f"Expected DuckDBBackend override, got {type(backend_override).__name__}"
            raise TypeError(message)
        response = backend_override.list_subsystems(
            limit=ctx.params.get_int("limit", default=0),
            role=ctx.params.get_str("role"),
            q=ctx.params.get_str("query"),
        )
        return CliResult.ok(
            SubsystemListResult(
                subsystems=[row.model_dump() for row in response.subsystems],
                meta=response.meta.model_dump(),
            )
        )

    ctx.logger.debug("Listing subsystems")
    backend = _build_backend(ctx)

    response = backend.list_subsystems(
        limit=ctx.params.get_int("limit"),
        role=ctx.params.get_str("role"),
        q=ctx.params.get_str("query"),
    )

    return CliResult.ok(
        SubsystemListResult(
            subsystems=[row.model_dump() for row in response.subsystems],
            meta=response.meta.model_dump(),
        )
    )


def subsystem_show_handler(ctx: CommandContext) -> CliResult[SubsystemShowResult]:
    """Show subsystem detail and modules.

    Parameters
    ----------
    ctx
        Command context with params:

        - subsystem_id: Subsystem ID to show (required)

    Returns
    -------
    CliResult[SubsystemShowResult]
        Result with subsystem details.
    """
    subsystem_id = ctx.params.require_str("subsystem_id")
    ctx.logger.debug("Showing subsystem: %s", subsystem_id)

    backend = _build_backend(ctx)
    response = backend.get_subsystem_modules(subsystem_id=subsystem_id)

    if not response.found or response.subsystem is None:
        LOG.debug("Subsystem not found: %s", subsystem_id)
        return fail_subsystem_not_found(subsystem_id)

    return CliResult.ok(
        SubsystemShowResult(
            subsystem=response.subsystem.model_dump(),
            modules=[row.model_dump() for row in response.modules],
            meta=response.meta.model_dump(),
        )
    )


def subsystem_profiles_handler(ctx: CommandContext) -> CliResult[SubsystemProfilesResult]:
    """List subsystem profiles from docs.v_subsystem_profile.

    Parameters
    ----------
    ctx
        Command context with optional params:

        - limit: Optional result limit

    Returns
    -------
    CliResult[SubsystemProfilesResult]
        Result with profile list.
    """
    ctx.logger.debug("Listing subsystem profiles")
    backend = _build_backend(ctx)

    response = backend.service.list_subsystem_profiles(limit=ctx.params.get_int("limit"))
    profile_response = (
        response
        if isinstance(response, SubsystemProfileResponse)
        else SubsystemProfileResponse.from_domain(response)
    )
    profiles = [
        row if isinstance(row, SubsystemProfileRow) else SubsystemProfileRow.model_validate(row)
        for row in profile_response.profiles
    ]

    return CliResult.ok(
        SubsystemProfilesResult(
            profiles=[row.model_dump() for row in profiles],
            meta=profile_response.meta.model_dump(),
        )
    )


def subsystem_coverage_handler(ctx: CommandContext) -> CliResult[SubsystemCoverageResult]:
    """List subsystem coverage rollups from docs.v_subsystem_coverage.

    Parameters
    ----------
    ctx
        Command context with optional params:

        - limit: Optional result limit

    Returns
    -------
    CliResult[SubsystemCoverageResult]
        Result with coverage list.
    """
    ctx.logger.debug("Listing subsystem coverage")
    backend = _build_backend(ctx)

    response = backend.service.list_subsystem_coverage(limit=ctx.params.get_int("limit"))
    coverage_response = (
        response
        if isinstance(response, SubsystemCoverageResponse)
        else SubsystemCoverageResponse.from_domain(response)
    )
    coverage_rows = [
        row if isinstance(row, SubsystemCoverageRow) else SubsystemCoverageRow.model_validate(row)
        for row in coverage_response.coverage
    ]

    return CliResult.ok(
        SubsystemCoverageResult(
            coverage=[row.model_dump() for row in coverage_rows],
            meta=coverage_response.meta.model_dump(),
        )
    )


def subsystem_module_memberships_handler(
    ctx: CommandContext,
) -> CliResult[SubsystemMembershipResult]:
    """List subsystem memberships for a module.

    Parameters
    ----------
    ctx
        Command context with params:

        - module: Module path to query (required)

    Returns
    -------
    CliResult[SubsystemMembershipResult]
        Result with membership list.
    """
    module = ctx.params.require_str("module")
    ctx.logger.debug("Getting subsystem memberships for module: %s", module)

    backend = _build_backend(ctx)
    response = backend.get_module_subsystems(module=module)

    return CliResult.ok(
        SubsystemMembershipResult(
            found=response.found,
            memberships=[row.model_dump() for row in response.memberships],
            meta=response.meta.model_dump(),
        )
    )


__all__ = [
    "SubsystemCoverageResult",
    "SubsystemListResult",
    "SubsystemMembershipResult",
    "SubsystemProfilesResult",
    "SubsystemShowResult",
    "subsystem_coverage_handler",
    "subsystem_list_handler",
    "subsystem_module_memberships_handler",
    "subsystem_profiles_handler",
    "subsystem_show_handler",
]
