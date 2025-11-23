"""Shared ingestion utilities (schema guards, iteration, logging, inserts)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from codeintel.config.schemas.sql_builder import PREPARED, ensure_schema
from codeintel.utils.paths import normalize_rel_path

log = logging.getLogger(__name__)

# Progress logging cadence defaults
PROGRESS_LOG_INTERVAL = 5.0
PROGRESS_LOG_EVERY = 50


@dataclass(frozen=True)
class ModuleRecord:
    """Metadata for a module path discovered in core.modules."""

    rel_path: str
    module_name: str
    file_path: Path
    index: int
    total: int


@dataclass(frozen=True)
class BatchResult:
    """Result metadata for a batch insert."""

    table_key: str
    rows: int
    duration_s: float


def log_progress(op: str, *, scope: str, table: str, rows: int, duration_s: float) -> None:
    """Structured ingest log entry."""
    log.info(
        "%s scope=%s table=%s rows=%d duration=%.2fs",
        op,
        scope,
        table,
        rows,
        duration_s,
    )


def load_module_map(
    con: duckdb.DuckDBPyConnection,
    repo: str,
    commit: str,
    *,
    language: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """
    Load path->module mapping from core.modules.

    Returns
    -------
    dict[str, str]
        Normalized mapping of relative path -> module name.
    """
    params: list[object] = [repo, commit]
    query = """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """
    if language is not None:
        query += " AND language = ?"
        params.append(language)
    rows = con.execute(query, params).fetchall()
    module_map = {normalize_rel_path(str(path)): str(module) for path, module in rows}
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
    Iterate modules with normalized paths and periodic progress logging.

    Returns
    -------
    Iterator[ModuleRecord]
        Iterator yielding module metadata for ingestion.
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
    con: duckdb.DuckDBPyConnection,
    table_key: str,
    rows: Sequence[Sequence[object]],
    *,
    delete_params: Sequence[object] | None = None,
    scope: str | None = None,
) -> BatchResult:
    """
    Ensure schema, delete prior rows (optional), insert batch, and log.

    Returns
    -------
    BatchResult
        Summary of rows inserted and elapsed time.
    """
    ensure_schema(con, table_key)
    stmts = PREPARED[table_key]

    start = time.perf_counter()
    if delete_params is not None and stmts.delete_sql is not None:
        con.execute(stmts.delete_sql, delete_params)
    if rows:
        con.executemany(stmts.insert_sql, rows)
    duration = time.perf_counter() - start

    if scope is not None:
        log_progress("ingest", scope=scope, table=table_key, rows=len(rows), duration_s=duration)
    else:
        log.info("ingest table=%s rows=%d duration=%.2fs", table_key, len(rows), duration)

    return BatchResult(table_key=table_key, rows=len(rows), duration_s=duration)


def insert_relation(
    con: duckdb.DuckDBPyConnection,
    table_key: str,
    rows: Sequence[Sequence[object]],
    *,
    delete_params: Sequence[object] | None = None,
    scope: str | None = None,
) -> BatchResult:
    """
    Insert rows via a temporary relation to avoid large VALUES clauses.

    Parameters
    ----------
    con:
        Live DuckDB connection.
    table_key:
        Registry table key (e.g., "core.ast_nodes").
    rows:
        Sequence of row tuples/lists matching registry column order.
    delete_params:
        Optional parameters for delete statement if present.
    scope:
        Optional repo@commit string for structured logging.

    Returns
    -------
    BatchResult
        Summary of rows inserted and elapsed time.
    """
    return run_batch(
        con,
        table_key,
        rows,
        delete_params=delete_params,
        scope=scope,
    )


def should_skip_empty(module_map: dict[str, str], *, logger: logging.Logger | None = None) -> bool:
    """
    Return True (and log) when no modules are present.

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
    """
    Return True (and log) when a required file is missing.

    Returns
    -------
    bool
        True if the file is missing, otherwise False.
    """
    if path.is_file():
        return False
    (logger or log).warning("%s not found; skipping ingestion", label)
    return True
