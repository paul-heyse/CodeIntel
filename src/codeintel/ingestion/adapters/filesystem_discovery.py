"""Filesystem discovery adapter implementing ModuleDiscoveryPort.

This adapter provides file system-based module discovery using the
existing SourceScanner infrastructure.
"""

from __future__ import annotations

import fnmatch
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.infrastructure.paths import relpath_to_module, repo_relpath
from codeintel.ingestion.infrastructure.scanning import SourceScanner
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.ingestion.infrastructure.scanning import ScanProfile

log = logging.getLogger(__name__)


PROGRESS_LOG_INTERVAL = 5.0
PROGRESS_LOG_EVERY = 50


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

        paths = list(scanner.iter_files())
        total = len(paths)

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
    def iter_modules(
        module_map: dict[str, str],
        repo_root: Path,
        *,
        logger: logging.Logger | None = None,
        scan_profile: ScanProfile | None = None,
    ) -> Iterator[ModuleRecord]:
        """Iterate modules with normalized paths and periodic progress logging.

        Parameters
        ----------
        module_map
            Mapping of relative paths to module names.
        repo_root
            Repository root directory.
        logger
            Optional logger for progress messages.
        scan_profile
            Optional scan profile for filtering.

        Returns
        -------
        Iterator[ModuleRecord]
            Iterator yielding module metadata for ingestion.
        """
        if not module_map:
            return iter(())

        active_log = logger or log
        patterns = tuple(scan_profile.include_globs) if scan_profile is not None else ("*",)
        ignore_set = set(scan_profile.ignore_dirs) if scan_profile is not None else set()
        log_every = scan_profile.log_every if scan_profile is not None else PROGRESS_LOG_EVERY
        log_interval = (
            scan_profile.log_interval if scan_profile is not None else PROGRESS_LOG_INTERVAL
        )
        filtered_items: list[tuple[str, str]] = []
        for rel_path, module_name in module_map.items():
            parts = Path(rel_path).parts
            if any(part in ignore_set for part in parts):
                continue
            if not any(fnmatch.fnmatch(rel_path, pat) for pat in patterns):
                continue
            filtered_items.append((rel_path, module_name))

        total = len(filtered_items)
        if total == 0:
            return iter(())
        start_ts = time.perf_counter()
        last_log = start_ts

        def _gen() -> Iterator[ModuleRecord]:
            nonlocal last_log
            for idx, (rel_path, module_name) in enumerate(filtered_items, start=1):
                file_path = repo_root / rel_path
                now_ts = time.perf_counter()
                if idx % log_every == 0 or (now_ts - last_log) >= log_interval:
                    elapsed = now_ts - start_ts
                    active_log.info("Module iteration %d/%d (%.2fs elapsed)", idx, total, elapsed)
                    last_log = now_ts
                yield ModuleRecord(
                    rel_path=rel_path,
                    module_name=module_name,
                    file_path=file_path,
                    index=idx,
                    total=total,
                )

        return _gen()

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


__all__ = [
    "PROGRESS_LOG_EVERY",
    "PROGRESS_LOG_INTERVAL",
    "FilesystemDiscoveryAdapter",
]
