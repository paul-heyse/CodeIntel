"""Shared ingestion utilities (schema guards, iteration, logging, inserts).

This module provides backward-compatible utilities for ingestion. The
underlying I/O operations now delegate to the port-adapter architecture
introduced in the ingestion refactoring.

Pure types and utilities are defined here; I/O operations delegate to adapters.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters.duckdb_storage import DuckDBStorageAdapter
from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter

# Re-export types from ports for backward compatibility
from codeintel.ingestion.ports.change_detection import ChangeRequest, ChangeSet
from codeintel.ingestion.ports.discovery import ModuleRecord
from codeintel.ingestion.ports.storage import BatchResult

if TYPE_CHECKING:
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

# Progress logging cadence defaults
PROGRESS_LOG_INTERVAL = 5.0
PROGRESS_LOG_EVERY = 50


class ChangeLogFileHandler(logging.FileHandler):
    """File handler tagged for change-detection logging.

    Attributes
    ----------
    codeintel_change_log
        Flag indicating this is a change log handler.
    """

    codeintel_change_log: bool
    _codeintel_change_log: bool

    def __init__(self, filename: str) -> None:
        """Initialize the handler.

        Parameters
        ----------
        filename
            Path to log file.
        """
        super().__init__(filename, encoding="utf-8")
        self.codeintel_change_log = True
        self._codeintel_change_log = True


def log_progress(op: str, *, scope: str, table: str, rows: int, duration_s: float) -> None:
    """Emit a structured ingest log entry.

    Parameters
    ----------
    op
        Operation name (e.g., "ingest").
    scope
        Scope identifier (e.g., "repo@commit").
    table
        Table name.
    rows
        Number of rows processed.
    duration_s
        Duration in seconds.
    """
    log.info(
        "%s scope=%s table=%s rows=%d duration=%.2fs",
        op,
        scope,
        table,
        rows,
        duration_s,
    )


def _get_change_logger() -> logging.Logger:
    """Return a logger that also writes to a file when configured.

    Set CODEINTEL_CHANGE_LOG to a file path to enable persistent logging
    of change detection decisions.

    Returns
    -------
    logging.Logger
        Logger configured for change detection diagnostics.
    """
    logger = logging.getLogger("codeintel.ingestion.change")
    logger.setLevel(logging.INFO)
    log_path = os.getenv("CODEINTEL_CHANGE_LOG")
    if log_path:
        existing = any(isinstance(handler, ChangeLogFileHandler) for handler in logger.handlers)
        if not existing:
            handler = ChangeLogFileHandler(log_path)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = True
    return logger


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
    log_interval = scan_profile.log_interval if scan_profile is not None else PROGRESS_LOG_INTERVAL
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


def read_module_source(record: ModuleRecord, *, logger: logging.Logger | None = None) -> str | None:
    """Read module source text with UTF-8 decoding.

    Parameters
    ----------
    record
        Module record with file path.
    logger
        Optional logger for warnings.

    Returns
    -------
    str | None
        Source text when readable; None when missing or undecodable.
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


def run_batch(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[Sequence[object]],
    *,
    delete_params: Sequence[object] | None = None,
    scope: str | None = None,
) -> BatchResult:
    """Ensure schema, delete prior rows (optional), insert batch, and log.

    This function delegates to `DuckDBStorageAdapter` for the actual I/O
    operations, maintaining backward compatibility while using the new
    port-adapter architecture.

    Parameters
    ----------
    gateway
        StorageGateway providing DuckDB access.
    table_key
        Registry table key (e.g., "core.ast_nodes").
    rows
        Row payload matching the prepared insert statement.
    delete_params
        Optional parameters for the delete statement when defined.
    scope
        Optional repo@commit string for structured logging.

    Returns
    -------
    BatchResult
        Summary of rows inserted and elapsed time.
    """
    adapter = DuckDBStorageAdapter(gateway)

    # Handle delete if parameters provided
    if delete_params is not None:
        adapter.delete_by_params(table_key, delete_params)

    # Write batch using adapter
    return adapter.write_batch(table_key, rows, scope=scope)


def insert_relation(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[Sequence[object]],
    *,
    delete_params: Sequence[object] | None = None,
    scope: str | None = None,
) -> BatchResult:
    """Insert rows via a temporary relation to avoid large VALUES clauses.

    This is an alias for `run_batch` maintained for backward compatibility.

    Parameters
    ----------
    gateway
        StorageGateway providing access to the DuckDB connection.
    table_key
        Registry table key (e.g., "core.ast_nodes").
    rows
        Sequence of row tuples/lists matching registry column order.
    delete_params
        Optional parameters for delete statement if present.
    scope
        Optional repo@commit string for structured logging.

    Returns
    -------
    BatchResult
        Summary of rows inserted and elapsed time.
    """
    return run_batch(
        gateway,
        table_key,
        rows,
        delete_params=delete_params,
        scope=scope,
    )


def should_skip_empty(module_map: dict[str, str], *, logger: logging.Logger | None = None) -> bool:
    """Return True (and log) when no modules are present.

    Parameters
    ----------
    module_map
        Module mapping to check.
    logger
        Optional logger for warning.

    Returns
    -------
    bool
        True if module_map is empty, otherwise False.
    """
    if module_map:
        return False
    (logger or log).warning("Skipping ingestion: module map is empty")
    return True


def should_skip_missing_file(
    path: Path, *, logger: logging.Logger | None = None, label: str
) -> bool:
    """Return True (and log) when a required file is missing.

    Parameters
    ----------
    path
        Path to check.
    logger
        Optional logger for warning.
    label
        Label for the missing file in log message.

    Returns
    -------
    bool
        True if the file is missing, otherwise False.
    """
    if path.is_file():
        return False
    (logger or log).warning("%s not found; skipping ingestion", label)
    return True


def compute_changes(
    gateway: StorageGateway,
    request: ChangeRequest,
) -> ChangeSet:
    """Compute changes for the given change request.

    This function provides backward compatibility by wrapping the
    HashChangeDetectionAdapter.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    request
        Change detection request parameters.

    Returns
    -------
    ChangeSet
        Computed changes (added, modified, deleted modules).
    """
    storage = DuckDBStorageAdapter(gateway)
    adapter = HashChangeDetectionAdapter(storage)
    # Get modules from request if available
    modules = getattr(request, "modules", []) or []
    return adapter.compute_changes(request, modules)


__all__ = [
    "PROGRESS_LOG_EVERY",
    "PROGRESS_LOG_INTERVAL",
    "BatchResult",
    "ChangeLogFileHandler",
    "ChangeRequest",
    "ChangeSet",
    "ModuleRecord",
    "compute_changes",
    "insert_relation",
    "iter_modules",
    "log_progress",
    "read_module_source",
    "run_batch",
    "should_skip_empty",
    "should_skip_missing_file",
]
