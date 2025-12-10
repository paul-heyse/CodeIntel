"""Typer-free handlers for subsystem exploration commands.

.. deprecated:: 2.0
    This module is deprecated. Use :mod:`codeintel.cli.handlers.subsystem` instead.
    This module will be removed in version 3.0.

These helpers keep operational logic while allowing Cyclopts to invoke
them without importing Typer. All user-facing errors surface as
:class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.runtime import build_graph_runtime as build_graph_runtime_internal
from codeintel.cli.cli_errors import ValidationError

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
    build_project_runtime,
    find_project_root,
)
from codeintel.cli.results import CliResult
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

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)

# Emit deprecation warning on import
warnings.warn(
    "codeintel.cli.subsystem_handlers is deprecated. Use codeintel.cli.handlers.subsystem instead.",
    DeprecationWarning,
    stacklevel=2,
)


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


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
# Result Types
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Structured Handler Helper
# -----------------------------------------------------------------------------


def _build_backend_from_ctx(ctx: ExecutionContext) -> DuckDBBackend:
    """Build a DuckDBBackend from execution context.

    Parameters
    ----------
    ctx
        Execution context.

    Returns
    -------
    DuckDBBackend
        Constructed backend instance.

    Raises
    ------
    RuntimeError
        If project cannot be resolved.
    TypeError
        If resolved backend is not DuckDBBackend.
    """
    setup_logging(ctx.verbosity)

    project_root_raw = ctx.params.get("project_root")
    project_root = Path(project_root_raw) if project_root_raw else None

    try:
        project_root_resolved = find_project_root(project_root)
        runtime = build_project_runtime(project_root_resolved)
    except ProjectNotFoundError as exc:
        msg = f"Project not found: {exc}"
        raise RuntimeError(msg) from exc
    except Exception as exc:
        msg = f"Failed to load project: {exc}"
        raise RuntimeError(msg) from exc

    gateway = open_gateway_from_config(runtime.cfg, read_only=True)
    graph_runtime = build_graph_runtime(runtime.cfg, gateway)

    resource = build_backend_resource(
        runtime.serving,
        gateway=gateway,
        options=BackendResourceOptions(graph_runtime=graph_runtime),
    )

    backend = resource.backend
    if not isinstance(backend, DuckDBBackend):
        msg = "Expected DuckDBBackend for local_db mode"
        raise TypeError(msg)
    return backend


# -----------------------------------------------------------------------------
# Structured Handlers (accept ExecutionContext)
# -----------------------------------------------------------------------------


def subsystem_list_ctx(ctx: ExecutionContext) -> CliResult[SubsystemListResult]:
    """List inferred subsystems with role/risk metadata.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - role: Optional role filter.
        - query: Optional search query.
        - limit: Optional result limit.

    Returns
    -------
    CliResult[SubsystemListResult]
        Result with subsystem list.
    """
    backend = _build_backend_from_ctx(ctx)
    response = backend.list_subsystems(
        limit=ctx.params.get("limit"),
        role=ctx.get_str_param("role"),
        q=ctx.get_str_param("query"),
    )
    return CliResult.ok(
        SubsystemListResult(
            subsystems=[row.model_dump() for row in response.subsystems],
            meta=response.meta.model_dump(),
        )
    )


def subsystem_show_ctx(ctx: ExecutionContext) -> CliResult[SubsystemShowResult]:
    """Show subsystem detail and modules.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - subsystem_id: Subsystem ID to show.

    Returns
    -------
    CliResult[SubsystemShowResult]
        Result with subsystem details.

    Raises
    ------
    RuntimeError
        If the subsystem cannot be found.
    """
    backend = _build_backend_from_ctx(ctx)
    subsystem_id = ctx.require_str_param("subsystem_id")
    response = backend.get_subsystem_modules(subsystem_id=subsystem_id)

    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", subsystem_id)
        msg = f"Subsystem not found: {subsystem_id}"
        raise RuntimeError(msg)

    return CliResult.ok(
        SubsystemShowResult(
            subsystem=response.subsystem.model_dump(),
            modules=[row.model_dump() for row in response.modules],
            meta=response.meta.model_dump(),
        )
    )


def subsystem_profiles_ctx(ctx: ExecutionContext) -> CliResult[SubsystemProfilesResult]:
    """List subsystem profiles from docs.v_subsystem_profile.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - limit: Optional result limit.

    Returns
    -------
    CliResult[SubsystemProfilesResult]
        Result with profile list.
    """
    backend = _build_backend_from_ctx(ctx)
    response = backend.service.list_subsystem_profiles(limit=ctx.params.get("limit"))
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


def subsystem_coverage_ctx(ctx: ExecutionContext) -> CliResult[SubsystemCoverageResult]:
    """List subsystem coverage rollups from docs.v_subsystem_coverage.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - limit: Optional result limit.

    Returns
    -------
    CliResult[SubsystemCoverageResult]
        Result with coverage list.
    """
    backend = _build_backend_from_ctx(ctx)
    response = backend.service.list_subsystem_coverage(limit=ctx.params.get("limit"))
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


def subsystem_module_memberships_ctx(ctx: ExecutionContext) -> CliResult[SubsystemMembershipResult]:
    """List subsystem memberships for a module.

    Parameters
    ----------
    ctx
        Execution context with params:
        - project_root: Optional project root override.
        - module: Module path to query.

    Returns
    -------
    CliResult[SubsystemMembershipResult]
        Result with membership list.
    """
    backend = _build_backend_from_ctx(ctx)
    module = ctx.require_str_param("module")
    response = backend.get_module_subsystems(module=module)
    return CliResult.ok(
        SubsystemMembershipResult(
            found=response.found,
            memberships=[row.model_dump() for row in response.memberships],
            meta=response.meta.model_dump(),
        )
    )


__all__ = [
    "RuntimeCliOptions",
    "SubsystemCoverageOptions",
    "SubsystemCoverageResult",
    "SubsystemIdOptions",
    "SubsystemListOptions",
    "SubsystemListResult",
    "SubsystemMembershipOptions",
    "SubsystemMembershipResult",
    "SubsystemProfilesOptions",
    "SubsystemProfilesResult",
    "SubsystemRuntime",
    "SubsystemShowResult",
    "build_graph_runtime",
    "build_runtime_from_cli",
    "open_gateway_from_config",
    "setup_logging",
    "subsystem_coverage_ctx",
    "subsystem_list_ctx",
    "subsystem_module_memberships_ctx",
    "subsystem_profiles_ctx",
    "subsystem_show_ctx",
]
