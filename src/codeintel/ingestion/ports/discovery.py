"""Module discovery port protocol for source file enumeration.

This module defines the port protocol for discovering source modules in a
repository. The protocol abstracts filesystem operations to enable testing
with virtual file systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.source_scanner import ScanProfile


@dataclass(frozen=True)
class ModuleRecord:
    """Metadata for a discovered source module.

    Attributes
    ----------
    rel_path
        Path relative to repository root.
    module_name
        Python module name (dot-separated).
    file_path
        Absolute path to the file.
    index
        Position in iteration (1-based).
    total
        Total number of modules being processed.
    """

    rel_path: str
    module_name: str
    file_path: Path
    index: int
    total: int


@runtime_checkable
class ModuleDiscoveryPort(Protocol):
    """Port protocol for discovering and reading source modules.

    This protocol abstracts filesystem operations to enable testing with
    mock file systems and to centralize module enumeration logic.
    """

    def discover_modules(
        self,
        repo_root: Path,
        profile: ScanProfile,
    ) -> Sequence[ModuleRecord]:
        """Discover all source modules matching the scan profile.

        Parameters
        ----------
        repo_root
            Repository root directory.
        profile
            Scan profile controlling file discovery.

        Returns
        -------
        Sequence[ModuleRecord]
            Discovered modules with metadata.
        """
        ...

    def read_module_source(self, record: ModuleRecord) -> str | None:
        """Read the source code of a module.

        Parameters
        ----------
        record
            Module record with file path.

        Returns
        -------
        str | None
            Source code text if readable, None if file is missing or undecodable.
        """
        ...

    def file_exists(self, path: Path) -> bool:
        """Check if a file exists.

        Parameters
        ----------
        path
            Path to check.

        Returns
        -------
        bool
            True if file exists.
        """
        ...

    def read_text(self, path: Path, encoding: str = "utf-8") -> str | None:
        """Read text content from a file.

        Parameters
        ----------
        path
            Path to read.
        encoding
            Text encoding.

        Returns
        -------
        str | None
            File content if readable, None otherwise.
        """
        ...


__all__ = [
    "ModuleDiscoveryPort",
    "ModuleRecord",
]
