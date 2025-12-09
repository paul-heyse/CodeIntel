"""Typer-free shared utilities for CLI command handlers.

This module provides common functionality used across all CLI handler modules,
including configuration building, gateway management, logging setup, and
type-safe option handling.

All user-facing errors surface as :class:`~codeintel.cli.cli_errors.ValidationError`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from codeintel.analytics.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.analytics.runtime import build_graph_runtime as build_graph_runtime_internal
from codeintel.cli.cli_errors import ValidationError
from codeintel.cli.cli_types import (
    BackendFlags,
    OutputFormat,
    PathSelection,
    RepoSelection,
    RuntimeOptions,
)

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.project import (
    ProjectConfig,
    ProjectNotFoundError,
    ProjectRuntime,
    StorageProjectConfig,
    build_project_runtime,
)
from codeintel.config import GraphRunScope
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.gateway_cache import close_gateways, get_gateway

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


# Backward compatible alias - RuntimeCliOptions is now an alias for RuntimeOptions
# from cli_types.py. This maintains backward compatibility with existing code.
RuntimeCliOptions = RuntimeOptions


@dataclass(frozen=True)
class RuntimeSelection:
    """Aggregated runtime inputs split by domain."""

    project_root: Path | None
    repo: RepoSelection
    paths: PathSelection
    backend: BackendFlags = field(default_factory=BackendFlags)


@dataclass(frozen=True)
class GatewayOptions:
    """Gateway usage preferences."""

    read_only: bool = True
    use_cache: bool = True


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


# -----------------------------------------------------------------------------
# Flag Resolution
# -----------------------------------------------------------------------------


def resolve_flag(value: object) -> bool:
    """Resolve an optional flag value to a boolean.

    Parameters
    ----------
    value
        Flag value (may be None, bool, or other).

    Returns
    -------
    bool
        True if value is truthy and not None, False otherwise.
    """
    if value is None:
        return False
    return bool(value)


# -----------------------------------------------------------------------------
# Configuration Building
# -----------------------------------------------------------------------------


def build_graph_backend_config(flags: BackendFlags) -> GraphBackendConfig:
    """Build graph backend configuration from CLI options.

    Parameters
    ----------
    flags
        Backend preferences collected from CLI flags.

    Returns
    -------
    GraphBackendConfig
        Configured graph backend settings.
    """
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    if flags.backend == "cpu":
        backend = "cpu"
    elif flags.backend == "nx-cugraph":
        backend = "nx-cugraph"
    return GraphBackendConfig(
        use_gpu=flags.use_gpu,
        backend=backend,
        strict=flags.strict,
    )


def _parse_env_flag(value: str | None, *, default: bool | None = None) -> bool | None:
    """Parse a boolean-ish environment string.

    Parameters
    ----------
    value
        Environment variable value.
    default
        Default value if parsing fails.

    Returns
    -------
    bool | None
        Parsed boolean or default.
    """
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_graph_feature_flags_from_env() -> GraphFeatureFlags:
    """Construct GraphFeatureFlags from CODEINTEL_* environment variables.

    Returns
    -------
    GraphFeatureFlags
        Feature flags derived from environment variables.
    """
    eager = (
        _parse_env_flag(os.environ.get("CODEINTEL_GRAPH_EAGER"))
        if "CODEINTEL_GRAPH_EAGER" in os.environ
        else None
    )
    community_limit = (
        int(os.environ["CODEINTEL_GRAPH_COMMUNITY_LIMIT"])
        if "CODEINTEL_GRAPH_COMMUNITY_LIMIT" in os.environ
        else None
    )
    validation_strict = (
        _parse_env_flag(os.environ.get("CODEINTEL_GRAPH_VALIDATION_STRICT"))
        if "CODEINTEL_GRAPH_VALIDATION_STRICT" in os.environ
        else None
    )
    return GraphFeatureFlags(
        eager_hydration=eager,
        community_detection_limit=community_limit,
        validation_strict=validation_strict,
    )


def build_config_from_options(
    repo: str,
    commit: str,
    paths_cfg: CliPathsInput,
    backend: BackendFlags,
) -> CodeIntelConfig:
    """Build CodeIntelConfig from explicit CLI options.

    Parameters
    ----------
    repo
        Repository slug.
    commit
        Commit SHA.
    paths_cfg
        CLI paths input describing repo root, build directory, and storage.
    backend
        Graph backend flags captured from CLI.

    Returns
    -------
    CodeIntelConfig
        Configured CodeIntel settings.
    """
    graph_backend = build_graph_backend_config(backend)
    graph_features = build_graph_feature_flags_from_env()
    LOG.info(
        "cli.runtime.config repo=%s commit=%s backend=%s use_gpu=%s features=%s",
        repo,
        commit,
        graph_backend.backend,
        graph_backend.use_gpu,
        graph_features,
    )
    repo_cfg = RepoConfig(repo=repo, commit=commit)
    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(graph_backend=graph_backend, graph_features=graph_features),
    )


# -----------------------------------------------------------------------------
# Scope Parsing
# -----------------------------------------------------------------------------


def parse_scope_args(
    scope_paths: list[str] | None,
    scope_modules: list[str] | None,
    scope_time_start: str | None,
    scope_time_end: str | None,
) -> GraphRunScope | None:
    """Build GraphRunScope from CLI scope arguments.

    Parameters
    ----------
    scope_paths
        List of path filters.
    scope_modules
        List of module filters.
    scope_time_start
        Start of time window (ISO8601).
    scope_time_end
        End of time window (ISO8601).

    Returns
    -------
    GraphRunScope | None
        Scope override when any flags are set, None otherwise.

    Raises
    ------
    ValueError
        When time window is incomplete or cannot be parsed.
    """
    paths = tuple(scope_paths or ())
    modules = tuple(scope_modules or ())

    time_window: tuple[datetime, datetime] | None = None
    if scope_time_start is not None or scope_time_end is not None:
        if scope_time_start is None or scope_time_end is None:
            msg = "Both --scope-time-start and --scope-time-end are required for time filtering"
            raise ValueError(msg)
        start = datetime.fromisoformat(scope_time_start)
        end = datetime.fromisoformat(scope_time_end)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        time_window = (start, end)

    if not paths and not modules and time_window is None:
        return None
    return GraphRunScope(paths=paths, modules=modules, time_window=time_window)


# -----------------------------------------------------------------------------
# Runtime Building
# -----------------------------------------------------------------------------


def _runtime_selection_from_options(
    options: RuntimeSelection | RuntimeCliOptions,
) -> RuntimeSelection:
    """Convert RuntimeCliOptions to RuntimeSelection.

    Parameters
    ----------
    options
        Options to convert.

    Returns
    -------
    RuntimeSelection
        Converted selection.
    """
    if isinstance(options, RuntimeSelection):
        return options
    return RuntimeSelection(
        project_root=options.project_root,
        repo=RepoSelection(repo=options.repo, commit=options.commit),
        paths=PathSelection(
            repo_root=options.repo_root,
            db_path=options.db_path,
            build_dir=options.build_dir,
            document_output_dir=options.document_output_dir,
        ),
        backend=options.backend,
    )


def build_runtime_from_cli(
    options: RuntimeSelection | RuntimeCliOptions,
) -> ProjectRuntime:
    """Build project runtime with fallback to explicit options.

    Tries project file discovery first. Falls back to explicit options
    if project file not found and all required options are provided.

    Parameters
    ----------
    options
        Runtime discovery inputs and backend flags.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    ValidationError
        If configuration cannot be resolved.
    """
    selection = _runtime_selection_from_options(options)

    # Try project file discovery first
    try:
        return build_project_runtime(selection.project_root)
    except ProjectNotFoundError:
        pass

    # Check if we have explicit fallback options
    if selection.repo.repo is None or selection.repo.commit is None:
        msg = "No codeintel.yaml found. Provide --repo and --commit explicitly."
        raise ValidationError(msg)

    # Build from explicit options
    resolved_repo_root = selection.paths.repo_root or Path.cwd()
    resolved_db_path = selection.paths.db_path or Path("build/db/codeintel.duckdb")
    resolved_build_dir = selection.paths.build_dir or Path("build")
    paths_cfg = CliPathsInput(
        repo_root=resolved_repo_root,
        build_dir=resolved_build_dir,
        db_path=resolved_db_path,
        document_output_dir=selection.paths.document_output_dir,
    )

    cfg = build_config_from_options(
        repo=selection.repo.repo,
        commit=selection.repo.commit,
        paths_cfg=paths_cfg,
        backend=selection.backend,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths

    # Ensure database directory exists
    paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    storage_cfg = StorageConfig.for_ingest(db_path=paths.db_path)
    gateway = open_gateway(storage_cfg)

    # Build minimal project config for the runtime
    project = ProjectConfig(
        repo=selection.repo.repo,
        storage=StorageProjectConfig(db_path=resolved_db_path),
    )

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    return ProjectRuntime(
        root=resolved_repo_root,
        project=project,
        cfg=cfg,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=cfg.tools,
        serving=serving,
    )


# -----------------------------------------------------------------------------
# Gateway Management
# -----------------------------------------------------------------------------


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


def build_graph_runtime(cfg: CodeIntelConfig, gateway: StorageGateway) -> GraphRuntime:
    """Construct a GraphRuntime for CLI commands.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    gateway
        Storage gateway.

    Returns
    -------
    GraphRuntime
        Runtime bound to the CLI snapshot and backend settings.
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


def build_paths_from_cli(paths: CliPathsInput) -> BuildPaths:
    """Convert CLI paths input into BuildPaths used by ingestion.

    Parameters
    ----------
    paths
        CLI paths input configuration.

    Returns
    -------
    BuildPaths
        Normalized internal paths.
    """
    return paths.to_build_paths()


# -----------------------------------------------------------------------------
# Unified Command Context
# -----------------------------------------------------------------------------


@dataclass
class ProjectContext:
    """Unified context for CLI commands that need project access.

    This dataclass bundles the common requirements for project-aware commands,
    providing access to configuration, gateway, and snapshot information.
    """

    config: CodeIntelConfig
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths


def build_project_context(
    runtime_options: RuntimeSelection | RuntimeCliOptions,
    gateway_options: GatewayOptions | None = None,
) -> ProjectContext:
    """Build a unified project context for CLI commands.

    This function resolves project configuration and returns a context object
    that can be used by any CLI command needing access to the project.

    Parameters
    ----------
    runtime_options
        Runtime discovery inputs and backend flags.
    gateway_options
        Gateway usage preferences.

    Returns
    -------
    ProjectContext
        Unified project context.
    """
    resolved_gateway_options = gateway_options or GatewayOptions()
    runtime = build_runtime_from_cli(runtime_options)

    # Use the cached or non-cached gateway based on preference
    if resolved_gateway_options.use_cache:
        storage_cfg = (
            StorageConfig.for_readonly(runtime.paths.db_path)
            if resolved_gateway_options.read_only
            else StorageConfig.for_ingest(runtime.paths.db_path)
        )
        gateway = get_gateway(storage_cfg)
    else:
        gateway = runtime.gateway

    return ProjectContext(
        config=runtime.cfg,
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
    )


def resolve_gateway_for_command(
    cfg: CodeIntelConfig,
    gateway_options: GatewayOptions | None = None,
) -> StorageGateway:
    """Resolve a StorageGateway for a CLI command.

    Parameters
    ----------
    cfg
        CodeIntel configuration.
    gateway_options
        Gateway usage preferences.

    Returns
    -------
    StorageGateway
        Gateway ready for use.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)

    resolved_gateway_options = gateway_options or GatewayOptions()
    if resolved_gateway_options.use_cache:
        storage_cfg = (
            StorageConfig.for_readonly(cfg.paths.db_path)
            if resolved_gateway_options.read_only
            else StorageConfig.for_ingest(cfg.paths.db_path)
        )
        return get_gateway(storage_cfg)
    return open_gateway_from_config(cfg, read_only=resolved_gateway_options.read_only)


def cleanup_command_resources() -> None:
    """Clean up resources after CLI command execution.

    Call this at the end of command execution to release cached gateways
    and other resources.
    """
    close_gateways()


__all__ = [
    "LOG",
    "BackendFlags",
    "GatewayOptions",
    "OutputFormat",
    "PathSelection",
    "ProjectContext",
    "RepoSelection",
    "RuntimeCliOptions",
    "RuntimeSelection",
    "build_config_from_options",
    "build_graph_backend_config",
    "build_graph_feature_flags_from_env",
    "build_graph_runtime",
    "build_paths_from_cli",
    "build_project_context",
    "build_runtime_from_cli",
    "cleanup_command_resources",
    "open_gateway_from_config",
    "parse_scope_args",
    "resolve_flag",
    "resolve_gateway_for_command",
    "setup_logging",
]
