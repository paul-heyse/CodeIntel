"""Shared source scanning utilities for ingestion phases."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".tox",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass(frozen=True)
class ScanConfig:
    """Configuration for source scanning."""

    repo_root: Path
    include_patterns: tuple[str, ...] = ("*.py",)
    ignore_dirs: tuple[str, ...] = tuple(sorted(IGNORES))
    log_every: int = 50
    log_interval: float = 5.0


@dataclass(frozen=True)
class ScanRecord:
    """Single file discovered by the scanner."""

    path: Path
    rel_path: str
    index: int
    total: int


class SourceScanner:
    """Reusable scanner for Python sources with logging cadence."""

    def __init__(self, cfg: ScanConfig) -> None:
        self.cfg = cfg

    def iter_files(self, logger: logging.Logger) -> Iterator[ScanRecord]:
        """
        Yield ScanRecords for all source files matching the include patterns.

        Yields
        ------
        ScanRecord
            Discovered source file with relative path and progress info.
        """
        repo_root = self.cfg.repo_root
        patterns = self.cfg.include_patterns
        ignore_dirs = set(self.cfg.ignore_dirs)

        candidates: list[Path] = []
        search_root = repo_root / "src" if (repo_root / "src").is_dir() else repo_root
        for pattern in patterns:
            candidates.extend(search_root.rglob(pattern))

        filtered = []
        for path in candidates:
            rel_parts = path.relative_to(repo_root).parts
            if any(part in ignore_dirs for part in rel_parts):
                continue
            filtered.append(path)

        total = len(filtered)
        start_ts = time.perf_counter()
        last_log = start_ts

        for idx, path in enumerate(sorted(filtered), start=1):
            now_ts = time.perf_counter()
            if idx % self.cfg.log_every == 0 or (now_ts - last_log) >= self.cfg.log_interval:
                elapsed = now_ts - start_ts
                logger.info("Source scan %d/%d (%.2fs elapsed)", idx, total, elapsed)
                last_log = now_ts
            yield ScanRecord(
                path=path,
                rel_path=str(path.relative_to(repo_root)).replace("\\", "/"),
                index=idx,
                total=total,
            )
