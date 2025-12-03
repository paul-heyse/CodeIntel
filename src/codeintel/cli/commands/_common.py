"""Shared utilities and patterns for CLI command implementations.

This module provides common functionality used across all CLI command groups,
including configuration building, gateway management, logging setup, and
type-safe option handling.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from codeintel.cli.project import (
    ProjectNotFoundError,
    ProjectRuntime,
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
from codeintel.storage.config import StorageConfig
from codeintel.storage.gateway import StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.analytics.graph_runtime import GraphRuntime

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
# Common CLI Option Types
# -----------------------------------------------------------------------------

# Project discovery options
ProjectRootOpt = Annotated[
    Path | None,
    typer.Option("--root", "-r", help="Explicit project root directory"),
]

VerboseOpt = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (can be repeated: -v=INFO, -vv=DEBUG)",
    ),
]

JsonOutputOpt = Annotated[
    bool,
    typer.Option("--json", help="Output as JSON", is_flag=True),
]

# Repository configuration options (fallback when no project file)
RepoOpt = Annotated[
    str | None,
    typer.Option(
        "--repo",
        help="Repository slug (e.g., 'org/repo'). Uses codeintel.yaml if not specified.",
    ),
]

CommitOpt = Annotated[
    str | None,
    typer.Option(
        "--commit",
        help="Commit SHA. Auto-detected from git if not specified.",
    ),
]

DbPathOpt = Annotated[
    Path | None,
    typer.Option(
        "--db-path",
        help="Path to DuckDB database. Uses project config if not specified.",
    ),
]

BuildDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--build-dir",
        help="Build directory (default: build/)",
    ),
]

DocumentOutputDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--document-output-dir",
        help="Override Document Output/ directory",
    ),
]

RepoRootOpt = Annotated[
    Path | None,
    typer.Option(
        "--repo-root",
        help="Path to repository root (default: current directory)",
    ),
]

# Graph backend options
NxGpuOpt = Annotated[
    bool,
    typer.Option(
        "--nx-gpu",
        is_flag=True,
        help="Prefer GPU backend for NetworkX (nx-cugraph) when available.",
    ),
]

NxBackendOpt = Annotated[
    str,
    typer.Option(
        "--nx-backend",
        help="NetworkX backend selection: auto, cpu, or nx-cugraph (default: auto).",
    ),
]

NxGpuStrictOpt = Annotated[
    bool,
    typer.Option(
        "--nx-gpu-strict",
        is_flag=True,
        help="Fail instead of falling back to CPU if GPU backend unavailable.",
    ),
]

# Scope filtering options
ScopePathOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--scope-path",
        help="Limit graph metrics to relative paths (repeatable).",
    ),
]

ScopeModuleOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--scope-module",
        help="Limit graph metrics to module names (repeatable).",
    ),
]

ScopeTimeWindowStartOpt = Annotated[
    str | None,
    typer.Option(
        "--scope-time-start",
        help="Limit metrics to commits after this ISO8601 timestamp.",
    ),
]

ScopeTimeWindowEndOpt = Annotated[
    str | None,
    typer.Option(
        "--scope-time-end",
        help="Limit metrics to commits before this ISO8601 timestamp.",
    ),
]

# Limit options
LimitOpt = Annotated[
    int | None,
    typer.Option("--limit", "-n", help="Limit number of results"),
]


# -----------------------------------------------------------------------------
# Flag Resolution
# -----------------------------------------------------------------------------


def resolve_flag(value: object) -> bool:
    """Resolve an optional flag value to a boolean.

    Parameters
    ----------
    value
        Flag value from Typer (may be None, bool, or other).

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


def build_graph_backend_config(
    nx_gpu: bool = False,
    nx_backend: str = "auto",
    nx_gpu_strict: bool = False,
) -> GraphBackendConfig:
    """Build graph backend configuration from CLI options.

    Parameters
    ----------
    nx_gpu
        Whether to prefer GPU backend.
    nx_backend
        Backend selection string.
    nx_gpu_strict
        Whether to fail if GPU is unavailable.

    Returns
    -------
    GraphBackendConfig
        Configured graph backend settings.
    """
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    if nx_backend == "cpu":
        backend = "cpu"
    elif nx_backend == "nx-cugraph":
        backend = "nx-cugraph"
    return GraphBackendConfig(
        use_gpu=nx_gpu,
        backend=backend,
        strict=nx_gpu_strict,
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
    repo_root: Path,
    db_path: Path,
    build_dir: Path,
    document_output_dir: Path | None = None,
    nx_gpu: bool = False,
    nx_backend: str = "auto",
    nx_gpu_strict: bool = False,
) -> CodeIntelConfig:
    """Build CodeIntelConfig from explicit CLI options.

    Parameters
    ----------
    repo
        Repository slug.
    commit
        Commit SHA.
    repo_root
        Repository root path.
    db_path
        Database path.
    build_dir
        Build directory path.
    document_output_dir
        Optional document output directory.
    nx_gpu
        Whether to prefer GPU backend.
    nx_backend
        NetworkX backend selection.
    nx_gpu_strict
        Whether to fail if GPU unavailable.

    Returns
    -------
    CodeIntelConfig
        Configured CodeIntel settings.
    """
    graph_backend = build_graph_backend_config(nx_gpu, nx_backend, nx_gpu_strict)
    graph_features = build_graph_feature_flags_from_env()
    LOG.info(
        "cli.runtime.config repo=%s commit=%s backend=%s use_gpu=%s features=%s",
        repo,
        commit,
        graph_backend.backend,
        graph_backend.use_gpu,
        graph_features,
    )
    paths_cfg = CliPathsInput(
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db_path,
        document_output_dir=document_output_dir,
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


def build_runtime_or_exit(
    project_root: Path | None = None,
    repo: str | None = None,
    commit: str | None = None,
    db_path: Path | None = None,
    build_dir: Path | None = None,
    repo_root: Path | None = None,
    document_output_dir: Path | None = None,
    nx_gpu: bool = False,
    nx_backend: str = "auto",
    nx_gpu_strict: bool = False,
) -> ProjectRuntime:
    """Build project runtime with fallback to explicit options.

    Tries project file discovery first. Falls back to explicit options
    if project file not found and all required options are provided.

    Parameters
    ----------
    project_root
        Explicit project root for project file discovery.
    repo
        Fallback repository slug.
    commit
        Fallback commit SHA.
    db_path
        Fallback database path.
    build_dir
        Fallback build directory.
    repo_root
        Fallback repository root.
    document_output_dir
        Fallback document output directory.
    nx_gpu
        Whether to prefer GPU backend.
    nx_backend
        NetworkX backend selection.
    nx_gpu_strict
        Whether to fail if GPU unavailable.

    Returns
    -------
    ProjectRuntime
        Constructed runtime context.

    Raises
    ------
    typer.Exit
        If configuration cannot be resolved.
    """
    # Try project file discovery first
    try:
        return build_project_runtime(project_root)
    except ProjectNotFoundError:
        pass

    # Check if we have explicit fallback options
    if repo is None or commit is None:
        typer.secho(
            "Error: No codeintel.yaml found. Provide --repo and --commit explicitly.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    # Build from explicit options
    resolved_repo_root = repo_root or Path.cwd()
    resolved_db_path = db_path or Path("build/db/codeintel.duckdb")
    resolved_build_dir = build_dir or Path("build")

    cfg = build_config_from_options(
        repo=repo,
        commit=commit,
        repo_root=resolved_repo_root,
        db_path=resolved_db_path,
        build_dir=resolved_build_dir,
        document_output_dir=document_output_dir,
        nx_gpu=nx_gpu,
        nx_backend=nx_backend,
        nx_gpu_strict=nx_gpu_strict,
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

    from codeintel.cli.project import ProjectConfig, StorageProjectConfig
    from codeintel.config.serving_models import ServingConfig

    # Build minimal project config for the runtime
    project = ProjectConfig(
        repo=repo,
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

    from codeintel.cli.project import ProjectRuntime

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
    from codeintel.analytics.graph_runtime import (
        GraphRuntimeOptions,
    )
    from codeintel.analytics.graph_runtime import (
        build_graph_runtime as _build_graph_runtime,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    return _build_graph_runtime(
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


__all__ = [
    "LOG",
    "BuildDirOpt",
    "CommitOpt",
    "DbPathOpt",
    "DocumentOutputDirOpt",
    "JsonOutputOpt",
    "LimitOpt",
    "NxBackendOpt",
    "NxGpuOpt",
    "NxGpuStrictOpt",
    "ProjectRootOpt",
    "RepoOpt",
    "RepoRootOpt",
    "ScopeModuleOpt",
    "ScopePathOpt",
    "ScopeTimeWindowEndOpt",
    "ScopeTimeWindowStartOpt",
    "VerboseOpt",
    "build_config_from_options",
    "build_graph_backend_config",
    "build_graph_feature_flags_from_env",
    "build_graph_runtime",
    "build_paths_from_cli",
    "build_runtime_or_exit",
    "open_gateway_from_config",
    "parse_scope_args",
    "resolve_flag",
    "setup_logging",
]
