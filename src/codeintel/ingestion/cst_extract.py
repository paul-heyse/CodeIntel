"""Extract LibCST concrete syntax trees into DuckDB tables."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import libcst as cst
from libcst import metadata

from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord, read_module_source, run_batch
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


def _resolve_worker_count(max_workers: int | None = None) -> int:
    """
    Determine thread pool size with an environment override.

    Returns
    -------
    int
        Worker count respecting CODEINTEL_CST_WORKERS when set; otherwise a
        conservative default based on host CPU.
    """
    if max_workers is not None and max_workers > 0:
        return max_workers
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


def _executor(kind: str, max_workers: int) -> ThreadPoolExecutor | ProcessPoolExecutor:
    """
    Return an executor for the requested backend.

    Parameters
    ----------
    kind
        Either "thread" (default) or "process".
    max_workers
        Parallelism to use.

    Returns
    -------
    ThreadPoolExecutor | ProcessPoolExecutor
        Executor matching the requested backend.
    """
    if kind == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    return ThreadPoolExecutor(max_workers=max_workers)


def ingest_cst(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
    executor_kind: str | None = None,
) -> None:
    """
    Parse modules listed in core.modules using LibCST and populate cst_nodes.

    Parameters
    ----------
    tracker:
        Shared change tracker providing access to change sets and gateway.
    max_workers:
        Optional pool size override for CST parsing.
    executor_kind:
        Optional override for executor selection ("thread" or "process").
    """
    worker_count = _resolve_worker_count(max_workers)
    exec_kind = (executor_kind or os.getenv("CODEINTEL_CST_EXECUTOR", "process")).lower()
    ops = CstIngestOps(
        repo=tracker.change_request.repo,
        commit=tracker.change_request.commit,
        executor_kind=exec_kind,
    )

    def _executor_factory() -> ThreadPoolExecutor | ProcessPoolExecutor:
        return _executor(exec_kind, worker_count)

    run_incremental_ingest(tracker, ops, executor_factory=_executor_factory)


def _delete_existing_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_paths: list[str],
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
                repo,
                commit,
                rel_paths,
            ],
        )
        return
    run_batch(gateway, "core.cst_nodes", [], delete_params=[repo, commit])


class CstIngestOps(IncrementalIngestOps[ModuleResult]):
    """Incremental ingest operations for LibCST extraction."""

    dataset_name = "core.cst_nodes"

    def __init__(self, *, repo: str, commit: str, executor_kind: str) -> None:
        self.repo = repo
        self.commit = commit
        self.executor_kind = executor_kind

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Check whether the module should be ingested for CST extraction.

        Parameters
        ----------
        module
            Module metadata describing the candidate file.

        Returns
        -------
        bool
            True when the module points to a Python source file.
        """
        return module.rel_path.endswith(".py")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        Remove CST rows for modules scheduled for deletion.

        Parameters
        ----------
        gateway
            Gateway used to execute DELETE statements.
        rel_paths
            Relative paths whose rows should be removed.
        """
        if not rel_paths:
            return
        _delete_existing_rows(
            gateway,
            repo=self.repo,
            commit=self.commit,
            rel_paths=list(rel_paths),
        )

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[ModuleResult]:
        """
        Parse a module and return serialized CST rows.

        Parameters
        ----------
        module
            Module metadata describing the file to parse.

        Returns
        -------
        list[ModuleResult]
            Extraction result containing rows or parse errors.
        """
        return [_process_module(module)]

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[ModuleResult]) -> None:
        """
        Flush CST rows into DuckDB in bounded batches.

        Parameters
        ----------
        gateway
            Gateway whose connection accepts batched inserts.
        rows
            Module extraction results yielded from workers.
        """
        batch: list[Row] = []
        inserted_rows = 0
        for result in rows:
            if result.error is not None:
                log.warning("Failed to parse %s: %s", result.rel_path, result.error)
            if result.rows:
                batch.extend(result.rows)
            if len(batch) >= FLUSH_EVERY:
                inserted_rows += len(batch)
                batch = _flush_batch(gateway, batch)

        if batch:
            inserted_rows += len(batch)
            _flush_batch(gateway, batch)
        log.info(
            "CST ingestion complete for %s@%s (rows=%d executor=%s)",
            self.repo,
            self.commit,
            inserted_rows,
            self.executor_kind,
        )


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
