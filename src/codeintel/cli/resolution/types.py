"""Resolution result types for the CLI resolution layer.

This module defines the immutable result types returned by resolution operations.
The primary type is ResolvedRuntime, which contains all resolved project information
needed by CLI handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.project import ProjectConfig
    from codeintel.config.models import CodeIntelConfig, ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.config.serving_models import ServingConfig


@dataclass(frozen=True)
class ResolvedRuntime:
    """Fully resolved runtime - the immutable result of runtime resolution.

    This dataclass contains all resolved project information needed by handlers.
    It is created by resolve_from_params() and cached in CommandContext.

    All fields are required and immutable (frozen=True).

    Parameters
    ----------
    root
        Project root directory.
    project
        Project configuration from codeintel.yaml or constructed from params.
    snapshot
        Repository snapshot reference (repo, commit, repo_root).
    paths
        Resolved build paths (db_path, build_dir, etc.).
    config
        Full CodeIntel configuration.
    serving
        Serving configuration for API operations.

    Examples
    --------
    >>> from codeintel.cli.resolution import resolve_runtime
    >>> runtime = resolve_runtime(ctx)  # doctest: +SKIP
    >>> runtime.db_path  # doctest: +SKIP
    PosixPath('build/db/codeintel.duckdb')
    >>> runtime.repo  # doctest: +SKIP
    'org/repo'
    """

    root: Path
    project: ProjectConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    config: CodeIntelConfig
    serving: ServingConfig

    @property
    def db_path(self) -> Path:
        """Get database path shortcut.

        Returns
        -------
        Path
            Path to DuckDB database file.
        """
        return self.paths.db_path

    @property
    def repo(self) -> str:
        """Get repository slug shortcut.

        Returns
        -------
        str
            Repository slug (e.g., 'org/repo').
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Get commit SHA shortcut.

        Returns
        -------
        str
            Commit SHA.
        """
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Get repository root shortcut.

        Returns
        -------
        Path
            Repository root directory.
        """
        return self.snapshot.repo_root

    @property
    def tools(self) -> ToolsConfig:
        """Get tools configuration shortcut.

        Returns
        -------
        ToolsConfig
            Tools configuration for external binaries.
        """
        return self.config.tools


__all__ = ["ResolvedRuntime"]
