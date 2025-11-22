"""Shared helpers for ingestion tasks (module maps, iteration, logging)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# Progress logging cadence used across ingestion tasks
PROGRESS_LOG_INTERVAL = 5.0
PROGRESS_LOG_EVERY = 50


@dataclass
class ModuleRecord:
    """Metadata for a module path discovered in core.modules."""

    rel_path: str
    module_name: str
    file_path: Path
    index: int
    total: int


def load_module_map(
    con: duckdb.DuckDBPyConnection,
    repo: str,
    commit: str,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """
    Load the module->qualname map from core.modules.

    Returns
    -------
    dict[str, str]
        Mapping of relative file path to module name.
    """
    rows = con.execute(
        """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    module_map = dict(rows)
    if not module_map:
        (logger or log).warning("No modules found in core.modules for %s@%s", repo, commit)
    return module_map


def iter_modules(
    module_map: dict[str, str],
    repo_root: Path,
    *,
    logger: logging.Logger | None = None,
    log_every: int = PROGRESS_LOG_EVERY,
    log_interval: float = PROGRESS_LOG_INTERVAL,
) -> Iterator[ModuleRecord]:
    """
    Iterate modules with periodic progress logging.

    Returns
    -------
    Iterator[ModuleRecord]
        Iterator over discovered modules with progress logging.
    """
    if not module_map:
        return iter(())

    active_log = logger or log
    total = len(module_map)
    start_ts = time.perf_counter()
    last_log = start_ts

    def _gen() -> Iterator[ModuleRecord]:
        nonlocal last_log
        for idx, (rel_path, module_name) in enumerate(module_map.items(), start=1):
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


def read_module_source(record: ModuleRecord, *, logger: logging.Logger | None = None) -> str | None:
    """
    Read module source text with UTF-8 decoding.

    Returns
    -------
    str | None
        Source text, or None if the file is missing or undecodable.
    """
    active_log = logger or log
    if not record.file_path.is_file():
        active_log.warning("Module path missing on disk: %s", record.file_path)
        return None
    try:
        return record.file_path.read_text(encoding="utf8")
    except UnicodeDecodeError:
        active_log.warning("Failed to read %s as UTF-8; skipping", record.file_path)
        return None
