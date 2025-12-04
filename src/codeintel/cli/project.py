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
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from codeintel.config.models import CliPathsInput, CodeIntelConfig, RepoConfig, ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway

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


_GIT_SHA_LENGTH = 40
"""Length of a full Git SHA-1 commit hash."""

_GIT_PACKED_REF_PARTS = 2
"""Minimum number of parts in a packed-refs line."""


def _read_file_safe(path: Path) -> str | None:
    """Read file content safely, returning None on error.

    Parameters
    ----------
    path
        Path to the file to read.

    Returns
    -------
    str | None
        File content stripped, or None if file doesn't exist or can't be read.
    """
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _resolve_packed_ref(git_dir: Path, ref_path: str) -> str | None:
    """Resolve a reference from packed-refs file.

    Parameters
    ----------
    git_dir
        Path to the .git directory.
    ref_path
        The reference path to look up.

    Returns
    -------
    str | None
        Commit SHA if found, None otherwise.
    """
    packed_refs = _read_file_safe(git_dir / "packed-refs")
    if packed_refs is None:
        return None

    for line in packed_refs.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= _GIT_PACKED_REF_PARTS and parts[1] == ref_path:
            return parts[0]
    return None


def _read_git_head(git_dir: Path) -> str | None:
    """Read the HEAD commit from a git directory.

    Parameters
    ----------
    git_dir
        Path to the .git directory.

    Returns
    -------
    str | None
        The commit SHA if found, None otherwise.
    """
    head_content = _read_file_safe(git_dir / "HEAD")
    if head_content is None:
        return None

    # Direct commit reference (detached HEAD)
    if not head_content.startswith("ref:"):
        return head_content if len(head_content) == _GIT_SHA_LENGTH else None

    # Symbolic reference (e.g., "ref: refs/heads/main")
    ref_path = head_content[4:].strip()
    ref_content = _read_file_safe(git_dir / ref_path)
    if ref_content is not None:
        return ref_content

    # Fall back to packed-refs
    return _resolve_packed_ref(git_dir, ref_path)


def detect_commit(root: Path) -> str:
    """Detect current commit (best-effort).

    Tries CODEINTEL_COMMIT env, then reads .git/HEAD, then 'HEAD'.

    Parameters
    ----------
    root
        Repository root directory for git detection.

    Returns
    -------
    str
        Detected commit SHA or 'HEAD' as fallback.
    """
    env_commit = os.environ.get("CODEINTEL_COMMIT")
    if env_commit:
        return env_commit

    git_dir = root / ".git"
    if git_dir.is_dir():
        commit = _read_git_head(git_dir)
        if commit:
            return commit

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

    Notes
    -----
    This function may propagate ProjectNotFoundError or ProjectConfigError
    from the underlying find_project_root and load_project_config calls.
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
