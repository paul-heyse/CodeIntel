"""Shared ingestion utilities (schema guards, iteration, logging, inserts)."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from codeintel.config.schemas.sql_builder import PREPARED, ensure_schema
from codeintel.core.config import SnapshotConfig
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path

log = logging.getLogger(__name__)

# Progress logging cadence defaults
PROGRESS_LOG_INTERVAL = 5.0
PROGRESS_LOG_EVERY = 50
FileStateRow = tuple[str, str, str, str, int, int, str]


@dataclass(frozen=True)
class ModuleRecord:
    """Metadata for a module path discovered in core.modules."""

    rel_path: str
    module_name: str
    file_path: Path
    index: int
    total: int


class ChangeLogFileHandler(logging.FileHandler):
    """File handler tagged for change-detection logging."""

    codeintel_change_log: bool
    _codeintel_change_log: bool

    def __init__(self, filename: str) -> None:
        super().__init__(filename, encoding="utf-8")
        self.codeintel_change_log = True
        self._codeintel_change_log = True


@dataclass(frozen=True)
class BatchResult:
    """Result metadata for a batch insert."""

    table_key: str
    rows: int
    duration_s: float


@dataclass(frozen=True)
class ChangeSet:
    """Added/modified/deleted modules detected for an ingest cycle."""

    added: list[ModuleRecord]
    modified: list[ModuleRecord]
    deleted: list[ModuleRecord]


@dataclass(frozen=True)
class ChangeSummary:
    """Aggregated change metadata for structured logging."""

    previous_count: int
    matched_count: int
    added: Sequence[ModuleRecord]
    modified_details: Sequence[tuple[str, str, str]]
    deleted: Sequence[ModuleRecord]


@dataclass(frozen=True)
class ChangeRequest:
    """Inputs required to compute a change set."""

    repo: str
    commit: str
    repo_root: Path
    language: str = "python"
    full_rebuild: bool = False
    scan_profile: ScanProfile | None = None
    modules: Sequence[ModuleRecord] | None = None
    logger: logging.Logger | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SnapshotConfig,
        *,
        scan_profile: ScanProfile,
        full_rebuild: bool = False,
        modules: Sequence[ModuleRecord] | None = None,
        logger: logging.Logger | None = None,
    ) -> ChangeRequest:
        """
        Build a ChangeRequest using a normalized snapshot configuration.

        Parameters
        ----------
        snapshot
            Snapshot describing repo root, slug, and commit.
        scan_profile
            Scan profile controlling source discovery.
        full_rebuild
            Whether to force a full ingest instead of incremental mode.
        modules
            Optional explicit module list to override filesystem scanning.
        logger
            Optional logger used for change-detection diagnostics.

        Returns
        -------
        ChangeRequest
            Normalized request bound to the provided snapshot.
        """
        return cls(
            repo=snapshot.repo_slug,
            commit=snapshot.commit,
            repo_root=snapshot.repo_root,
            language="python",
            full_rebuild=full_rebuild,
            scan_profile=scan_profile,
            modules=modules,
            logger=logger,
        )


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


def _get_change_logger() -> logging.Logger:
    """
    Return a logger that also writes to a file when configured.

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


def _file_digest(path: Path) -> tuple[int, int, str]:
    """
    Compute a stable digest for a file.

    Returns
    -------
    tuple[int, int, str]
        Size in bytes, mtime in nanoseconds, and a blake2b hex digest.
    """
    stat_result = path.stat()
    digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
    return stat_result.st_size, stat_result.st_mtime_ns, digest


def _build_current_state(
    records: Sequence[ModuleRecord],
    *,
    logger: logging.Logger,
) -> dict[str, tuple[ModuleRecord, int, int, str]]:
    """
    Compute file digests for the provided records.

    Returns
    -------
    dict[str, tuple[ModuleRecord, int, int, str]]
        Mapping of normalized rel_path to (record, size, mtime_ns, digest).
    """
    current: dict[str, tuple[ModuleRecord, int, int, str]] = {}
    for record in records:
        try:
            size, mtime_ns, digest = _file_digest(record.file_path)
        except OSError as exc:
            logger.warning("Failed to stat %s: %s", record.file_path, exc)
            continue
        normalized_rel = normalize_rel_path(record.rel_path)
        current[normalized_rel] = (record, size, mtime_ns, digest)
    return current


def _load_previous_state(
    gateway: StorageGateway,
    request: ChangeRequest,
    *,
    logger: logging.Logger,
) -> dict[str, tuple[int, int, str]]:
    """
    Load the most recent file_state rows for the request.

    Returns
    -------
    dict[str, tuple[int, int, str]]
        Mapping of normalized rel_path to (size, mtime_ns, digest).
    """
    ensure_schema(gateway.con, "core.file_state")
    rows = gateway.con.execute(
        """
        WITH ranked AS (
            SELECT
                rel_path,
                size_bytes,
                mtime_ns,
                content_hash,
                ROW_NUMBER() OVER (PARTITION BY rel_path ORDER BY mtime_ns DESC) AS rn
            FROM core.file_state
            WHERE repo = ? AND language = ?
        )
        SELECT rel_path, size_bytes, mtime_ns, content_hash
        FROM ranked
        WHERE rn = 1
        """,
        [request.repo, request.language],
    ).fetchall()
    previous = {
        normalize_rel_path(str(rel_path)): (int(size), int(mtime_ns), str(content_hash))
        for rel_path, size, mtime_ns, content_hash in rows
    }
    if not previous:
        logger.info("No previous file_state rows found for %s@%s", request.repo, request.commit)
    return previous


def _diff_states(
    current: dict[str, tuple[ModuleRecord, int, int, str]],
    previous: dict[str, tuple[int, int, str]],
    request: ChangeRequest,
    module_map: dict[str, str],
) -> tuple[list[ModuleRecord], list[ModuleRecord], list[tuple[str, str, str]], list[ModuleRecord]]:
    """
    Compute added, modified, and deleted modules between snapshots.

    Returns
    -------
    tuple[list[ModuleRecord], list[ModuleRecord], list[tuple[str, str, str]], list[ModuleRecord]]
        Added modules, modified modules, modified detail tuples, and deleted modules.
    """
    added: list[ModuleRecord] = []
    modified: list[ModuleRecord] = []
    modified_details: list[tuple[str, str, str]] = []
    for rel_path, (record, _size, _mtime_ns, digest) in current.items():
        old = previous.get(rel_path)
        if old is None:
            added.append(record)
        elif old[2] != digest:
            modified.append(record)
            modified_details.append((rel_path, old[2], digest))

    deleted: list[ModuleRecord] = [
        ModuleRecord(
            rel_path=rel_path,
            module_name=module_map.get(rel_path, "<deleted>"),
            file_path=request.repo_root / rel_path,
            index=0,
            total=0,
        )
        for rel_path in previous.keys() - current.keys()
    ]
    return added, modified, modified_details, deleted


def compute_changes(
    gateway: StorageGateway,
    request: ChangeRequest,
) -> ChangeSet:
    """
    Detect added/modified/deleted modules and refresh core.file_state.

    This function always updates the file_state snapshot for the provided
    repo/commit/language so subsequent runs compare against the latest state.

    Returns
    -------
    ChangeSet
        Detected additions, modifications, and deletions.
    """
    active_log = request.logger or log
    module_map = (
        {record.rel_path: record.module_name for record in request.modules}
        if request.modules is not None
        else load_module_map(
            gateway,
            request.repo,
            request.commit,
            language=request.language,
            logger=active_log,
        )
    )
    if should_skip_empty(module_map, logger=active_log):
        return ChangeSet(added=[], modified=[], deleted=[])

    records: Sequence[ModuleRecord]
    if request.modules is not None:
        records = request.modules
    else:
        records = tuple(
            iter_modules(
                module_map,
                request.repo_root,
                logger=active_log,
                scan_profile=request.scan_profile,
            )
        )

    current = _build_current_state(records, logger=active_log)
    previous = _load_previous_state(gateway, request, logger=active_log)
    added, modified, modified_details, deleted = _diff_states(
        current,
        previous,
        request,
        module_map,
    )

    file_state_rows: list[FileStateRow] = [
        (
            request.repo,
            request.commit,
            rel_path,
            request.language,
            size,
            mtime_ns,
            digest,
        )
        for rel_path, (_, size, mtime_ns, digest) in sorted(current.items())
    ]
    _upsert_file_state(gateway, request, file_state_rows)
    matched_count = sum(1 for rel in current if rel in previous)
    summary = ChangeSummary(
        previous_count=len(previous),
        matched_count=matched_count,
        added=added,
        modified_details=modified_details,
        deleted=deleted,
    )
    change_set = ChangeSet(added=added, modified=modified, deleted=deleted)
    active_log.info(
        "compute_changes repo=%s lang=%s prev_rows=%d matched=%d added=%d modified=%d deleted=%d",
        request.repo,
        request.language,
        summary.previous_count,
        summary.matched_count,
        len(change_set.added),
        len(change_set.modified),
        len(change_set.deleted),
    )
    _log_change_details(request, summary)
    return change_set


def _upsert_file_state(
    gateway: StorageGateway,
    request: ChangeRequest,
    file_state_rows: Sequence[FileStateRow],
) -> None:
    """Replace file_state rows for provided rel_paths."""
    ensure_schema(gateway.con, "core.file_state")
    if not file_state_rows:
        return
    rel_paths = [row[2] for row in file_state_rows]
    delete_params = [(request.repo, rel_path, request.language) for rel_path in rel_paths]
    gateway.con.executemany(
        "DELETE FROM core.file_state WHERE repo = ? AND rel_path = ? AND language = ?",
        delete_params,
    )
    gateway.con.executemany(PREPARED["core.file_state"].insert_sql, file_state_rows)


def _log_change_details(request: ChangeRequest, summary: ChangeSummary) -> None:
    """Persist change detection details to the configured change logger."""
    logger = _get_change_logger()
    logger.info(
        "compute_changes repo=%s lang=%s prev_rows=%d matched=%d added=%d modified=%d deleted=%d",
        request.repo,
        request.language,
        summary.previous_count,
        summary.matched_count,
        len(summary.added),
        len(summary.modified_details),
        len(summary.deleted),
    )
    max_samples = 20
    if summary.added:
        logger.info(
            "added_paths (first %d of %d): %s",
            max_samples,
            len(summary.added),
            [record.rel_path for record in summary.added[:max_samples]],
        )
    if summary.modified_details:
        logger.info(
            "modified_paths (first %d of %d): %s",
            max_samples,
            len(summary.modified_details),
            [
                f"{rel} {old_hash}->{new_hash}"
                for rel, old_hash, new_hash in summary.modified_details[:max_samples]
            ],
        )
    if summary.deleted:
        logger.info(
            "deleted_paths (first %d of %d): %s",
            max_samples,
            len(summary.deleted),
            [record.rel_path for record in summary.deleted[:max_samples]],
        )


def load_module_map(
    gateway: StorageGateway,
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
    con = gateway.con
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
    scan_profile: ScanProfile | None = None,
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
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[Sequence[object]],
    *,
    delete_params: Sequence[object] | None = None,
    scope: str | None = None,
) -> BatchResult:
    """
    Ensure schema, delete prior rows (optional), insert batch, and log.

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
    active_con = gateway.con
    ensure_schema(active_con, table_key)
    stmts = PREPARED[table_key]

    start = time.perf_counter()
    if delete_params is not None and stmts.delete_sql is not None:
        active_con.execute(stmts.delete_sql, delete_params)
    if rows:
        active_con.executemany(stmts.insert_sql, rows)
    duration = time.perf_counter() - start

    if scope is not None:
        log_progress("ingest", scope=scope, table=table_key, rows=len(rows), duration_s=duration)
    else:
        log.info("ingest table=%s rows=%d duration=%.2fs", table_key, len(rows), duration)

    return BatchResult(table_key=table_key, rows=len(rows), duration_s=duration)


def insert_relation(
    gateway: StorageGateway,
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
    gateway:
        StorageGateway providing access to the DuckDB connection or a raw connection.
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
        gateway,
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
