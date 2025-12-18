"""Resolution result types for the CLI resolution layer.

This module defines the immutable result types returned by resolution operations.
The primary type is ResolvedRuntime, which contains all resolved project information
needed by CLI handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.cli.project import ProjectConfig
    from codeintel.config.models import CodeIntelConfig, ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.core.runtime import RuntimePrimitives
    from codeintel.serving.config import ServingConfig


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
    primitives
        Canonical runtime primitive bundle (snapshot/paths/tools/graph config).
    config
        Full CodeIntel configuration.
    serving
        Serving configuration for API operations.

    Examples
    --------
    >>> from codeintel.cli.resolution import resolve_runtime
    >>> runtime = resolve_runtime(ctx)
    >>> runtime.db_path
    PosixPath('build/db/codeintel.duckdb')
    >>> runtime.repo
    'org/repo'
    """

    root: Path
    project: ProjectConfig
    primitives: RuntimePrimitives
    config: CodeIntelConfig
    serving: ServingConfig

    @property
    def snapshot(self) -> SnapshotRef:
        """Return snapshot identity for this runtime.

        Returns
        -------
        SnapshotRef
            Repository snapshot reference.
        """
        return self.primitives.snapshot

    @property
    def paths(self) -> BuildPaths:
        """Return build paths for this runtime.

        Returns
        -------
        BuildPaths
            Derived build path bundle.
        """
        return self.primitives.paths

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
