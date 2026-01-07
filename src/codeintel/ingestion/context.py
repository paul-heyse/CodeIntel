"""Shared ingestion context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    from codeintel.core.runtime import RuntimeSettings
    from codeintel.core.tools import ToolBinaries
    from codeintel.ingestion.infrastructure.scanning import ScanProfile


@dataclass(frozen=True, slots=True)
class IngestionContext:
    """Bundle common ingestion inputs to reduce parameter sprawl.

    Parameters
    ----------
    snapshot
        Snapshot reference containing repo, commit, and repo root.
    repo_root
        Repository root path.
    scan_profile
        Scan profile for module discovery.
    tools
        Tool binaries configuration.
    settings
        Runtime settings for ingestion configuration.
    """

    snapshot: SnapshotRef
    repo_root: Path
    scan_profile: ScanProfile
    tools: ToolBinaries
    settings: RuntimeSettings

    @property
    def repo(self) -> str:
        """Return repository identifier from snapshot."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier from snapshot."""
        return self.snapshot.commit


def resolve_repo_commit(
    *,
    context: IngestionContext | None,
    repo: str | None,
    commit: str | None,
) -> tuple[str, str]:
    """Resolve repo/commit from context or explicit parameters.

    Returns
    -------
    tuple[str, str]
        Resolved repository identifier and commit hash.

    Raises
    ------
    ValueError
        If repo/commit are missing and no context is provided.
    """
    if context is not None:
        return context.repo, context.commit
    if repo is None or commit is None:
        msg = "repo and commit are required when ingestion context is missing"
        raise ValueError(msg)
    return repo, commit


def resolve_repo_root(
    *,
    context: IngestionContext | None,
    repo_root: Path | None,
) -> Path:
    """Resolve repo root from context or explicit parameter.

    Returns
    -------
    Path
        Resolved repository root path.

    Raises
    ------
    ValueError
        If repo_root is missing and no context is provided.
    """
    if context is not None:
        return context.repo_root
    if repo_root is None:
        msg = "repo_root is required when ingestion context is missing"
        raise ValueError(msg)
    return repo_root


def resolve_scan_profile(
    *,
    context: IngestionContext | None,
    scan_profile: ScanProfile | None,
) -> ScanProfile:
    """Resolve scan profile from context or explicit parameter.

    Returns
    -------
    ScanProfile
        Resolved scan profile configuration.

    Raises
    ------
    ValueError
        If scan_profile is missing and no context is provided.
    """
    if context is not None:
        return context.scan_profile
    if scan_profile is None:
        msg = "scan_profile is required when ingestion context is missing"
        raise ValueError(msg)
    return scan_profile


def resolve_tools(
    *,
    context: IngestionContext | None,
    tools: ToolBinaries | None,
) -> ToolBinaries:
    """Resolve tool binaries from context or explicit parameter.

    Returns
    -------
    ToolBinaries
        Resolved tool binaries configuration.

    Raises
    ------
    ValueError
        If tools are missing and no context is provided.
    """
    if context is not None:
        return context.tools
    if tools is None:
        msg = "tools are required when ingestion context is missing"
        raise ValueError(msg)
    return tools


def resolve_settings(
    *,
    context: IngestionContext | None,
    settings: RuntimeSettings | None,
) -> RuntimeSettings:
    """Resolve runtime settings from context or explicit parameter.

    Returns
    -------
    RuntimeSettings
        Resolved runtime settings configuration.

    Raises
    ------
    ValueError
        If settings are missing and no context is provided.
    """
    if context is not None:
        return context.settings
    if settings is None:
        msg = "settings are required when ingestion context is missing"
        raise ValueError(msg)
    return settings


__all__ = [
    "IngestionContext",
    "resolve_repo_commit",
    "resolve_repo_root",
    "resolve_scan_profile",
    "resolve_settings",
    "resolve_tools",
]
