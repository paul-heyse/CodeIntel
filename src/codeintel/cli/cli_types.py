"""Canonical CLI type definitions.

This module is the single source of truth for all CLI-related types.
Other modules should import from here rather than defining their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class OutputFormat(Enum):
    """Output rendering format for CLI commands."""

    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True)
class BackendFlags:
    """Backend preferences provided via CLI.

    Parameters
    ----------
    use_gpu
        Whether to attempt GPU acceleration.
    backend
        Backend selection (auto, cpu, or nx-cugraph).
    strict
        Whether to enforce strict backend compatibility.
    """

    use_gpu: bool = False
    backend: str = "auto"
    strict: bool = False


@dataclass(frozen=True)
class RuntimeOptions:
    """Unified runtime discovery and backend options.

    This is the canonical runtime options structure used across all CLI modules.

    Parameters
    ----------
    project_root
        Root directory for project discovery.
    repo
        Repository identifier.
    commit
        Commit SHA.
    db_path
        Path to the database file.
    build_dir
        Build output directory.
    repo_root
        Repository root path.
    document_output_dir
        Document output directory.
    backend
        Backend configuration flags.
    """

    project_root: Path | None = None
    repo: str | None = None
    commit: str | None = None
    db_path: Path | None = None
    build_dir: Path | None = None
    repo_root: Path | None = None
    document_output_dir: Path | None = None
    backend: BackendFlags = field(default_factory=BackendFlags)


@dataclass(frozen=True)
class RepoSelection:
    """Repository identification inputs.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    """

    repo: str | None
    commit: str | None


@dataclass(frozen=True)
class PathSelection:
    """Repository path inputs for storage and builds.

    Parameters
    ----------
    repo_root
        Repository root path.
    db_path
        Path to the database file.
    build_dir
        Build output directory.
    document_output_dir
        Document output directory.
    """

    repo_root: Path | None
    db_path: Path | None
    build_dir: Path | None
    document_output_dir: Path | None = None


# Type alias for help level
HelpLevel = Literal["brief", "full"]


__all__ = [
    "BackendFlags",
    "HelpLevel",
    "OutputFormat",
    "PathSelection",
    "RepoSelection",
    "RuntimeOptions",
]
