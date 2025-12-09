"""Typer-free handlers for subsystem exploration commands.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.runtime import build_graph_runtime as build_graph_runtime_internal
from codeintel.cli.cli_errors import ValidationError
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.config.models import CodeIntelConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.serving.bootstrap import BackendResourceOptions, build_backend_resource
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.mcp.models import (
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemProfileResponse,
    SubsystemProfileRow,
)
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Parameters
    ----------
    verbosity
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeCliOptions:
    """CLI options for runtime resolution."""

    project_root: Path | None = None


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


# -----------------------------------------------------------------------------
# Runtime Utilities
# -----------------------------------------------------------------------------


def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Build a ProjectRuntime from CLI options.

    Parameters
    ----------
    options
        CLI options containing project root.

    Returns
    -------
    ProjectRuntime
        Resolved project runtime.

    Raises
    ------
    ValidationError
        If the project cannot be resolved.
    """
    try:
        project_root = find_project_root(options.project_root)
        return build_project_runtime(project_root)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise ValidationError(msg) from exc
    except Exception as exc:
        msg = f"Failed to load project: {exc}"
        raise ValidationError(msg) from exc


def open_gateway_from_config(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """Open a StorageGateway from CodeIntelConfig.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    read_only
        Whether to open read-only.

    Returns
    -------
    StorageGateway
        Opened gateway.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = (
        StorageConfig.for_readonly(cfg.paths.db_path)
        if read_only
        else StorageConfig.for_ingest(cfg.paths.db_path)
    )
    gateway_cfg = replace(
        base_cfg,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return open_gateway(gateway_cfg)


def build_graph_runtime(
    cfg: CodeIntelConfig,
    gateway: StorageGateway,
) -> GraphRuntime:
    """Build a graph runtime from config and gateway.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    gateway
        Storage gateway.

    Returns
    -------
    GraphRuntime
        Graph runtime instance.
    """
    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    return build_graph_runtime_internal(
        gateway,
        GraphRuntimeOptions(
            snapshot=snapshot,
            backend=cfg.graph_backend,
            features=cfg.graph_features,
        ),
    )


def _build_backend(runtime: SubsystemRuntime) -> DuckDBBackend:
    """Build a DuckDBBackend from runtime options.

    Parameters
    ----------
    runtime
        Subsystem runtime options.

    Returns
    -------
    DuckDBBackend
        Constructed backend instance.

    Raises
    ------
    ValidationError
        If the resolved backend is not a DuckDBBackend.
    """
    setup_logging(runtime.verbose)

    runtime_cfg = build_runtime_from_cli(runtime.runtime_options)
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
        raise ValidationError(msg)
    return backend


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------


def subsystem_list_handler(options: SubsystemListOptions) -> None:
    """List inferred subsystems with role/risk metadata.

    Parameters
    ----------
    options
        List options.
    """
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

    Parameters
    ----------
    options
        Subsystem ID options.

    Raises
    ------
    ValidationError
        If the subsystem cannot be found.
    """
    backend = _build_backend(options.runtime)
    response = backend.get_subsystem_modules(subsystem_id=options.subsystem_id)

    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", options.subsystem_id)
        msg = f"Subsystem not found: {options.subsystem_id}"
        raise ValidationError(msg)

    payload = {
        "subsystem": response.subsystem.model_dump(),
        "modules": [row.model_dump() for row in response.modules],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")


def subsystem_profiles_handler(options: SubsystemProfilesOptions) -> None:
    """List subsystem profiles from docs.v_subsystem_profile.

    Parameters
    ----------
    options
        Profiles options.
    """
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
    """List subsystem coverage rollups from docs.v_subsystem_coverage.

    Parameters
    ----------
    options
        Coverage options.
    """
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
    """List subsystem memberships for a module.

    Parameters
    ----------
    options
        Membership options.
    """
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
    "RuntimeCliOptions",
    "SubsystemCoverageOptions",
    "SubsystemIdOptions",
    "SubsystemListOptions",
    "SubsystemMembershipOptions",
    "SubsystemProfilesOptions",
    "SubsystemRuntime",
    "build_graph_runtime",
    "build_runtime_from_cli",
    "open_gateway_from_config",
    "setup_logging",
    "subsystem_coverage_handler",
    "subsystem_list_handler",
    "subsystem_module_memberships_handler",
    "subsystem_profiles_handler",
    "subsystem_show_handler",
]
