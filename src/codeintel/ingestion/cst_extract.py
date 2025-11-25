"""Extract LibCST concrete syntax trees into DuckDB tables."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import libcst as cst
from libcst import metadata

from codeintel.ingestion.common import (
    ChangeRequest,
    ChangeSet,
    ModuleRecord,
    compute_changes,
    iter_modules,
    load_module_map,
    read_module_source,
    run_batch,
    should_skip_empty,
)
from codeintel.ingestion.cst_utils import CstCaptureConfig, CstCaptureVisitor
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)
FLUSH_EVERY = 10_000
DEFAULT_MAX_WORKERS = 16
Row = tuple[str, str, str, dict[str, list[int]], str, tuple[str, ...], tuple[str, ...]]
CST_CAPTURE_CONFIG = CstCaptureConfig(
    kinds=(
        cst.Module,
        cst.FunctionDef,
        ASYNC_FUNC_DEF,
        cst.ClassDef,
        cst.Assign,
        cst.AnnAssign,
        cst.AugAssign,
        cst.Import,
        cst.ImportFrom,
        cst.Call,
        cst.Return,
        cst.Raise,
        cst.Yield,
        cst.If,
        cst.Else,
        cst.For,
        cst.While,
        cst.With,
        cst.Try,
        cst.ExceptHandler,
        cst.Match,
    ),
    snippet_limit=200,
)


@dataclass(frozen=True)
class ModuleResult:
    """Rows and errors returned from processing a single module."""

    rel_path: str
    rows: list[Row]
    error: str | None = None


class CstVisitor(CstCaptureVisitor):
    """Collect CST rows using shared capture helpers."""

    def __init__(self, rel_path: str, module_name: str, source: str) -> None:
        super().__init__(rel_path, module_name, source, config=CST_CAPTURE_CONFIG)


def _flush_batch(
    gateway: StorageGateway,
    rows: list[Row],
) -> list[Row]:
    if not rows:
        return []
    row_count = len(rows)
    normalized_rows = [
        [
            rel_path,
            node_id,
            kind,
            span,
            snippet,
            list(parents),
            list(qnames),
        ]
        for rel_path, node_id, kind, span, snippet, parents, qnames in rows
    ]
    run_batch(
        gateway,
        "core.cst_nodes",
        normalized_rows,
        delete_params=None,
    )
    log.debug("Flushed %d CST rows", row_count)
    return []


def _resolve_worker_count() -> int:
    """
    Determine thread pool size with an environment override.

    Returns
    -------
    int
        Worker count respecting CODEINTEL_CST_WORKERS when set; otherwise a
        conservative default based on host CPU.
    """
    env_workers = os.getenv("CODEINTEL_CST_WORKERS")
    if env_workers:
        try:
            value = int(env_workers)
            if value > 0:
                return value
        except ValueError:
            log.warning("Ignoring invalid CODEINTEL_CST_WORKERS=%s", env_workers)

    cpu_count = os.cpu_count() or 1
    return min(DEFAULT_MAX_WORKERS, max(2, cpu_count // 2))


def ingest_cst(
    gateway: StorageGateway,
    target: ChangeRequest,
    change_set: ChangeSet | None = None,
) -> None:
    """
    Parse modules listed in core.modules using LibCST and populate cst_nodes.

    Parameters
    ----------
    gateway:
        StorageGateway providing access to the DuckDB database.
    target:
        Change request describing repo, commit, and scan configuration.
    change_set:
        Optional precomputed change set; when provided, only touched modules are
        re-indexed and deletions are applied per file.
    """
    repo_root = target.repo_root.resolve()
    module_map = load_module_map(
        gateway, target.repo, target.commit, language=target.language, logger=log
    )
    if should_skip_empty(module_map, logger=log):
        return
    active_change_set = change_set or compute_changes(
        gateway,
        ChangeRequest(
            repo=target.repo,
            commit=target.commit,
            repo_root=repo_root,
            scan_config=target.scan_config,
            language=target.language,
            logger=log,
        ),
    )
    to_reparse = list(active_change_set.added + active_change_set.modified)
    deleted_paths = [record.rel_path for record in active_change_set.deleted]
    if not to_reparse and not deleted_paths:
        log.info("No CST changes detected for %s@%s", target.repo, target.commit)
        return

    total_modules = len(to_reparse) if to_reparse else (0 if deleted_paths else len(module_map))
    log.info("Parsing CST for %d modules in %s@%s", total_modules, target.repo, target.commit)

    cst_values: list[Row] = []
    start_ts = time.perf_counter()
    inserted_rows = 0

    rel_paths = [record.rel_path for record in to_reparse + active_change_set.deleted]
    _delete_existing_rows(gateway, target, rel_paths)

    records: list[ModuleRecord]
    if to_reparse:
        records = to_reparse
    elif deleted_paths:
        records = []
    else:
        records = list(
            iter_modules(
                module_map,
                repo_root,
                logger=log,
                scan_config=target.scan_config,
            )
        )

    if records:
        worker_count = _resolve_worker_count()
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for result in pool.map(_process_module, records):
                if result.error is not None:
                    log.warning("Failed to parse %s: %s", result.rel_path, result.error)
                if result.rows:
                    cst_values.extend(result.rows)
                if len(cst_values) >= FLUSH_EVERY:
                    inserted_rows += len(cst_values)
                    cst_values = _flush_batch(gateway, cst_values)

        inserted_rows += len(cst_values)
        cst_values = _flush_batch(gateway, cst_values)
    duration = time.perf_counter() - start_ts
    log.info(
        "CST extraction complete for %s@%s (%d modules, %d rows, %.2fs)",
        target.repo,
        target.commit,
        total_modules,
        inserted_rows,
        duration,
    )


def _delete_existing_rows(
    gateway: StorageGateway, target: ChangeRequest, rel_paths: list[str]
) -> None:
    """Remove stale CST rows prior to incremental inserts."""
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM core.cst_nodes
            WHERE path IN (
                SELECT path FROM core.modules WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            )
            """,
            [
                target.repo,
                target.commit,
                rel_paths,
            ],
        )
        return
    run_batch(gateway, "core.cst_nodes", [], delete_params=[target.repo, target.commit])


def _process_module(record: ModuleRecord) -> ModuleResult:
    source = read_module_source(record, logger=None)
    if source is None:
        return ModuleResult(rel_path=record.rel_path, rows=[], error=None)
    try:
        wrapper = metadata.MetadataWrapper(
            cst.parse_module(source),
            unsafe_skip_copy=True,
        )
        visitor = CstVisitor(
            rel_path=record.rel_path, module_name=record.module_name, source=source
        )
        wrapper.visit(visitor)
    except (cst.ParserSyntaxError, ValueError, TypeError, RuntimeError) as exc:
        return ModuleResult(rel_path=record.rel_path, rows=[], error=str(exc))

    return ModuleResult(rel_path=record.rel_path, rows=visitor.rows, error=None)
