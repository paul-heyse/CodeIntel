"""Shared source scanning utilities for ingestion phases."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

log = logging.getLogger(__name__)

IGNORES: Final[tuple[str, ...]] = (
    ".DS_Store",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".svn",
    ".tox",
    ".vscode",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
)

DEFAULT_IGNORE_DIRS: Final[tuple[str, ...]] = (
    *IGNORES,
)


@dataclass(frozen=True)
class ScanConfig:
    """Deprecated scan configuration retained for compatibility."""

    repo_root: Path
    include_patterns: tuple[str, ...] = ("*.py",)
    ignore_dirs: tuple[str, ...] = DEFAULT_IGNORE_DIRS
    log_every: int = 250
    log_interval: float = 5.0

    def to_profile(self) -> "ScanProfile":
        """Convert to the canonical ScanProfile representation."""
        return ScanProfile(
            repo_root=self.repo_root,
            source_roots=(self.repo_root,),
            include_globs=self.include_patterns,
            ignore_dirs=self.ignore_dirs,
            log_every=self.log_every,
            log_interval=self.log_interval,
        )


@dataclass(frozen=True)
class ScanProfile:
    """Description of how to scan a repository for a category of files."""

    repo_root: Path
    source_roots: tuple[Path, ...]
    include_globs: tuple[str, ...]
    ignore_dirs: tuple[str, ...] = DEFAULT_IGNORE_DIRS
    log_every: int = 250
    log_interval: float = 5.0


def default_code_profile(repo_root: Path) -> ScanProfile:
    """
    Build the default profile for Python source files.

    Falls back to the repository root when no src/ directory exists.

    Returns
    -------
    ScanProfile
        Profile configured to scan Python sources under the repository.
    """
    src_root = repo_root / "src"
    roots = (src_root,) if src_root.is_dir() else (repo_root,)
    return ScanProfile(
        repo_root=repo_root,
        source_roots=roots,
        include_globs=("**/*.py",),
        ignore_dirs=DEFAULT_IGNORE_DIRS,
    )


def default_config_profile(repo_root: Path) -> ScanProfile:
    """
    Profile for configuration-style files across the repository.

    Returns
    -------
    ScanProfile
        Profile configured to scan common config formats.
    """
    return ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("**/*.toml", "**/*.ini", "**/*.cfg", "**/*.yaml", "**/*.yml"),
        ignore_dirs=DEFAULT_IGNORE_DIRS,
    )


def _split_env_list(raw: str) -> tuple[str, ...]:
    """Split comma/semicolon separated env values into a tuple of entries."""
    separators = (",", ";")
    normalized = raw
    for sep in separators:
        normalized = normalized.replace(sep, ",")
    return tuple(entry.strip() for entry in normalized.split(",") if entry.strip())


def profile_from_env(base: ScanProfile) -> ScanProfile:
    """
    Apply CODEINTEL_INCLUDE_PATTERNS and CODEINTEL_IGNORE_DIRS to a base profile.

    Environment variables use comma or semicolon separators.

    Returns
    -------
    ScanProfile
        Updated profile with environment overrides applied.
    """
    include = base.include_globs
    ignore = list(base.ignore_dirs)

    raw_include = os.getenv("CODEINTEL_INCLUDE_PATTERNS")
    if raw_include:
        include = _split_env_list(raw_include)

    raw_ignore = os.getenv("CODEINTEL_IGNORE_DIRS")
    if raw_ignore:
        seen = set(ignore)
        for entry in _split_env_list(raw_ignore):
            if entry not in seen:
                ignore.append(entry)
                seen.add(entry)

    return ScanProfile(
        repo_root=base.repo_root,
        source_roots=base.source_roots,
        include_globs=include,
        ignore_dirs=tuple(ignore),
        log_every=base.log_every,
        log_interval=base.log_interval,
    )


class SourceScanner:
    """Reusable scanner for repository files with progress logging."""

    def __init__(self, profile: ScanProfile) -> None:
        self.profile = profile

    def iter_files(self) -> Iterator[Path]:
        """
        Yield files matching include globs while respecting ignored directories.

        Yields
        ------
        Path
            Paths to files that satisfy the scan profile.
        """
        ignore_set = set(self.profile.ignore_dirs)
        yielded = 0
        start_ts = time.perf_counter()
        last_log = start_ts

        for root in self.profile.source_roots:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [name for name in dirnames if name not in ignore_set]
                for name in filenames:
                    path = Path(dirpath) / name
                    relative_parts = path.relative_to(self.profile.repo_root).parts
                    if any(part in ignore_set for part in relative_parts):
                        continue
                    if not _matches_any_glob(path, self.profile.include_globs):
                        continue
                    yielded += 1
                    now_ts = time.perf_counter()
                    if yielded % self.profile.log_every == 0 or (
                        now_ts - last_log
                    ) >= self.profile.log_interval:
                        elapsed = now_ts - start_ts
                        log.info("Source scan %d files (%.2fs elapsed)", yielded, elapsed)
                        last_log = now_ts
                    yield path


def _matches_any_glob(path: Path, patterns: Iterable[str]) -> bool:
    posix = path.as_posix()
    relative = path.name if "/" not in posix else posix
    return any(path.match(pattern) or Path(relative).match(pattern) for pattern in patterns)
