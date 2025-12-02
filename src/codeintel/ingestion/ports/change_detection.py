"""Change detection port protocol for incremental ingestion.

This module defines the port protocol for detecting changes between
repository snapshots. The protocol abstracts file hashing and state
persistence to enable testing with mock implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.source_scanner import ScanProfile


@dataclass(frozen=True)
class FileDigest:
    """Digest information for a single file.

    Attributes
    ----------
    size_bytes
        File size in bytes.
    mtime_ns
        Modification time in nanoseconds.
    content_hash
        Content hash (blake2b).
    """

    size_bytes: int
    mtime_ns: int
    content_hash: str


@dataclass
class ChangeSet:
    """Set of changes detected between snapshots.

    Attributes
    ----------
    added
        Modules that were added.
    modified
        Modules that were modified.
    deleted
        Modules that were deleted.
    """

    added: list[ModuleRecord] = field(default_factory=list)
    modified: list[ModuleRecord] = field(default_factory=list)
    deleted: list[ModuleRecord] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Return True if any changes detected.

        Returns
        -------
        bool
            True if there are changes.
        """
        return bool(self.added or self.modified or self.deleted)

    @property
    def total_changed(self) -> int:
        """Return total number of changed modules.

        Returns
        -------
        int
            Total count of added, modified, and deleted modules.
        """
        return len(self.added) + len(self.modified) + len(self.deleted)


@dataclass(frozen=True)
class ChangeRequest:
    """Request parameters for change detection.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    repo_root
        Repository root path.
    language
        Source language (default: "python").
    full_rebuild
        Force full rebuild mode.
    scan_profile
        Scan profile for file discovery.
    modules
        Optional explicit module list.
    """

    repo: str
    commit: str
    repo_root: Path
    language: str = "python"
    full_rebuild: bool = False
    scan_profile: ScanProfile | None = None
    modules: Sequence[ModuleRecord] | None = None


@runtime_checkable
class ChangeDetectionPort(Protocol):
    """Port protocol for detecting file changes between snapshots.

    This protocol abstracts change detection to enable testing with
    mock implementations and to centralize incremental update logic.
    """

    def compute_changes(
        self,
        request: ChangeRequest,
        current_modules: Sequence[ModuleRecord],
    ) -> ChangeSet:
        """Compute changes between previous and current state.

        Parameters
        ----------
        request
            Change detection request parameters.
        current_modules
            Current modules discovered in the repository.

        Returns
        -------
        ChangeSet
            Detected changes (added, modified, deleted).
        """
        ...

    def load_previous_state(
        self,
        repo: str,
        language: str,
    ) -> Mapping[str, FileDigest]:
        """Load the previous file state from storage.

        Parameters
        ----------
        repo
            Repository identifier.
        language
            Source language.

        Returns
        -------
        Mapping[str, FileDigest]
            Mapping from relative path to file digest.
        """
        ...

    def save_current_state(
        self,
        repo: str,
        commit: str,
        language: str,
        state: Mapping[str, FileDigest],
    ) -> None:
        """Save the current file state to storage.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        language
            Source language.
        state
            Mapping from relative path to file digest.
        """
        ...

    def compute_file_digest(self, path: Path) -> FileDigest | None:
        """Compute the digest for a single file.

        Parameters
        ----------
        path
            Path to the file.

        Returns
        -------
        FileDigest | None
            File digest if readable, None otherwise.
        """
        ...


__all__ = [
    "ChangeDetectionPort",
    "ChangeRequest",
    "ChangeSet",
    "FileDigest",
]
