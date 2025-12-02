"""Filesystem discovery adapter implementing ModuleDiscoveryPort.

This adapter provides file system-based module discovery using the
existing SourceScanner infrastructure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.paths import relpath_to_module, repo_relpath
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.source_scanner import SourceScanner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.source_scanner import ScanProfile

log = logging.getLogger(__name__)


class FilesystemDiscoveryAdapter:
    """Filesystem discovery adapter implementing ModuleDiscoveryPort.

    This adapter discovers Python modules by scanning the file system
    using configurable scan profiles.

    Parameters
    ----------
    repo_root
        Repository root directory for path resolution.
    """

    def __init__(self, repo_root: Path) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        repo_root
            Repository root directory.
        """
        self._repo_root = repo_root

    @staticmethod
    def discover_modules(
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
        scanner = SourceScanner(profile)
        modules: list[ModuleRecord] = []

        # First pass: collect all files
        paths = list(scanner.iter_files())
        total = len(paths)

        # Second pass: build module records
        for idx, path in enumerate(paths, start=1):
            rel_path = repo_relpath(repo_root, path)
            module_name = relpath_to_module(rel_path)
            modules.append(
                ModuleRecord(
                    rel_path=rel_path,
                    module_name=module_name,
                    file_path=path,
                    index=idx,
                    total=total,
                )
            )

        log.info("Discovered %d modules in %s", len(modules), repo_root)
        return modules

    @staticmethod
    def read_module_source(record: ModuleRecord) -> str | None:
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
        if not record.file_path.is_file():
            log.warning("Module path missing on disk: %s", record.file_path)
            return None
        try:
            return record.file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Failed to read %s as UTF-8; skipping", record.file_path)
            return None

    @staticmethod
    def file_exists(path: Path) -> bool:
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
        return path.is_file()

    @staticmethod
    def read_text(path: Path, encoding: str = "utf-8") -> str | None:
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
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding=encoding)
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("Failed to read %s: %s", path, exc)
            return None


__all__ = ["FilesystemDiscoveryAdapter"]
