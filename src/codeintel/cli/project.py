"""Project configuration discovery and runtime construction for the CLI.

This module provides:
- Pydantic models for parsing `codeintel.yaml` project configuration
- Functions for discovering the project root directory
- Factory for constructing a unified runtime context from project config

The project config file (`codeintel.yaml`) provides a declarative way to
configure CodeIntel for a repository, avoiding verbose CLI flags.

Example Project Config
----------------------
.. code-block:: yaml

    repo: github.com/org/my-repo
    default_profile: default
    ingest:
      recipe: builtin.default
    analytics:
      profile: full
    graphs:
      recipe: builtin.full
    storage:
      db_path: .codeintel/duckdb.db
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from codeintel.config.models import CliPathsInput, CodeIntelConfig, RepoConfig, ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.config import StorageConfig
from codeintel.storage.gateway import StorageGateway, open_gateway

LOG = logging.getLogger(__name__)

PROJECT_FILE = "codeintel.yaml"
"""Canonical name for the project configuration file."""


# -----------------------------------------------------------------------------
# Project Configuration Models
# -----------------------------------------------------------------------------


class IngestProjectConfig(BaseModel):
    """Ingestion configuration within a project file.

    Attributes
    ----------
    recipe
        Ingestion recipe name to use (e.g., "builtin.default").
    """

    recipe: str = "builtin.default"


class AnalyticsProjectConfig(BaseModel):
    """Analytics configuration within a project file.

    Attributes
    ----------
    profile
        Analytics profile name (reserved for future use).
    """

    profile: str = "full"


class GraphsProjectConfig(BaseModel):
    """Graphs configuration within a project file.

    Attributes
    ----------
    recipe
        Graphs recipe name to use (e.g., "builtin.full").
    """

    recipe: str = "builtin.full"


class StorageProjectConfig(BaseModel):
    """Storage configuration within a project file.

    Attributes
    ----------
    db_path
        Path to the DuckDB database file relative to project root.
    """

    db_path: Path = Path(".codeintel/duckdb.db")


class ProjectConfig(BaseModel):
    """Project-level configuration loaded from codeintel.yaml.

    This model represents the complete project configuration, providing
    defaults for all optional sections.

    Attributes
    ----------
    repo
        Repository slug (e.g., "github.com/org/repo").
    default_profile
        Default profile name for CLI commands.
    ingest
        Ingestion configuration section.
    analytics
        Analytics configuration section.
    graphs
        Graphs configuration section.
    storage
        Storage configuration section.
    """

    repo: str
    default_profile: str = "default"
    ingest: IngestProjectConfig = Field(default=IngestProjectConfig())
    analytics: AnalyticsProjectConfig = Field(default=AnalyticsProjectConfig())
    graphs: GraphsProjectConfig = Field(default=GraphsProjectConfig())
    storage: StorageProjectConfig = Field(default=StorageProjectConfig())


# -----------------------------------------------------------------------------
# Project Discovery
# -----------------------------------------------------------------------------


class ProjectNotFoundError(Exception):
    """Raised when a project file cannot be found."""


class ProjectConfigError(Exception):
    """Raised when a project file cannot be parsed."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from start (or CWD) to find codeintel.yaml.

    Parameters
    ----------
    start
        Starting directory for the search (defaults to current working directory).

    Returns
    -------
    Path
        Absolute path to the directory containing codeintel.yaml.

    Raises
    ------
    ProjectNotFoundError
        If no project file is found in the directory hierarchy.

    Examples
    --------
    >>> root = find_project_root(Path("/path/to/nested/dir"))
    >>> (root / "codeintel.yaml").exists()
    True
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / PROJECT_FILE
        if candidate.is_file():
            return parent
    message = f"Could not find {PROJECT_FILE} starting from {current}"
    raise ProjectNotFoundError(message)


def load_project_config(root: Path | None = None) -> ProjectConfig:
    """Load ProjectConfig from codeintel.yaml at the given root.

    Parameters
    ----------
    root
        Project root directory (defaults to result of find_project_root).

    Returns
    -------
    ProjectConfig
        Parsed and validated project configuration.

    Raises
    ------
    ProjectNotFoundError
        If the project file does not exist.
    ProjectConfigError
        If the project file cannot be parsed as YAML or fails validation.
    """
    resolved_root = root or find_project_root()
    path = resolved_root / PROJECT_FILE
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"Project file {PROJECT_FILE} not found at {resolved_root}"
        raise ProjectNotFoundError(msg) from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        msg = f"Failed to parse {PROJECT_FILE}: {exc}"
        raise ProjectConfigError(msg) from exc

    if raw is None:
        msg = f"Project file {PROJECT_FILE} is empty"
        raise ProjectConfigError(msg)

    try:
        return ProjectConfig.model_validate(raw)
    except Exception as exc:
        msg = f"Failed to validate {PROJECT_FILE}: {exc}"
        raise ProjectConfigError(msg) from exc


# -----------------------------------------------------------------------------
# Commit Detection
# -----------------------------------------------------------------------------


def detect_commit(root: Path) -> str:
    """Detect current commit (best-effort).

    Tries CODEINTEL_COMMIT env, then `git rev-parse HEAD`, then 'HEAD'.

    Parameters
    ----------
    root
        Repository root directory for git commands.

    Returns
    -------
    str
        Detected commit SHA or 'HEAD' as fallback.
    """
    env_commit = os.environ.get("CODEINTEL_COMMIT")
    if env_commit:
        return env_commit

    git_dir = root / ".git"
    if git_dir.exists():
        try:
            git_bin = shutil.which("git")
            if git_bin:
                result = subprocess.run(
                    [git_bin, "rev-parse", "HEAD"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return "HEAD"


# -----------------------------------------------------------------------------
# Project Runtime
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectRuntime:
    """Runtime wiring derived from project config and current repo state.

    This dataclass bundles all the configuration objects needed to run
    pipeline commands or serve operations for a project.

    Parameters
    ----------
    root
        Absolute path to project root.
    project
        Parsed project configuration.
    cfg
        CodeIntelConfig derived from project settings.
    snapshot
        Repository snapshot reference.
    paths
        Build paths for the project.
    gateway
        Storage gateway connected to the project database.
    tools
        Tools configuration for external binaries.
    serving
        Serving configuration for HTTP/MCP servers.
    """

    root: Path
    project: ProjectConfig
    cfg: CodeIntelConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    gateway: StorageGateway
    tools: ToolsConfig
    serving: ServingConfig


def build_project_runtime(root: Path | None = None) -> ProjectRuntime:
    """Build runtime context from project config and environment.

    Used by all CLI commands to construct SnapshotRef, BuildPaths,
    StorageGateway, ToolsConfig, and ServingConfig from the project file.

    Parameters
    ----------
    root
        Optional project root (defaults to discovery via find_project_root).

    Returns
    -------
    ProjectRuntime
        Complete runtime context for CLI operations.

    Raises
    ------
    ProjectNotFoundError
        If the project root cannot be found.
    ProjectConfigError
        If the project configuration is invalid.
    """
    resolved_root = find_project_root(root)
    project = load_project_config(resolved_root)

    commit = detect_commit(resolved_root)
    repo_cfg = RepoConfig(repo=project.repo, commit=commit)

    db_path = resolved_root / project.storage.db_path
    paths_cfg = CliPathsInput(
        repo_root=resolved_root,
        build_dir=resolved_root / ".codeintel",
        db_path=db_path,
        document_output_dir=None,
    )

    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=None,
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

    tools = cfg.tools

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    LOG.info(
        "project.runtime repo=%s commit=%s root=%s db=%s",
        project.repo,
        commit,
        resolved_root,
        paths.db_path,
    )

    return ProjectRuntime(
        root=resolved_root,
        project=project,
        cfg=cfg,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        serving=serving,
    )


__all__ = [
    "PROJECT_FILE",
    "AnalyticsProjectConfig",
    "GraphsProjectConfig",
    "IngestProjectConfig",
    "ProjectConfig",
    "ProjectConfigError",
    "ProjectNotFoundError",
    "ProjectRuntime",
    "StorageProjectConfig",
    "build_project_runtime",
    "detect_commit",
    "find_project_root",
    "load_project_config",
]
