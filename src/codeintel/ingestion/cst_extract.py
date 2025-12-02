"""Extract LibCST concrete syntax trees into DuckDB tables."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import libcst as cst
from libcst import metadata

from codeintel.ingestion.common import ModuleRecord, read_module_source, run_batch
from codeintel.ingestion.cst_utils import CstCaptureConfig, CstCaptureVisitor
from codeintel.ingestion.pipeline import (
    IngestPipeline,
    PipelineConfig,
    PipelineResult,
    execute_pipeline,
)
from codeintel.ingestion.workers import CST_WORKER_CONFIG, resolve_worker_count

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)
FLUSH_EVERY = 10_000

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
) -> int:
    """
    Flush CST rows to DuckDB.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    rows
        Rows to persist.

    Returns
    -------
    int
        Number of rows flushed.
    """
    if not rows:
        return 0
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
    return row_count


def _process_module(record: ModuleRecord) -> ModuleResult:
    """
    Parse a module and return CST rows.

    Parameters
    ----------
    record
        Module metadata describing the file to parse.

    Returns
    -------
    ModuleResult
        Extraction result containing rows or parse error.
    """
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


class CstPipeline:
    """Pipeline implementation for CST extraction."""

    def __init__(self, *, repo: str, commit: str) -> None:
        self._repo = repo
        self._commit = commit

    @property
    def dataset_name(self) -> str:
        """Return the dataset name for this pipeline."""
        return "core.cst_nodes"

    def module_filter(self, module: ModuleRecord) -> bool:
        """
        Determine whether CST extraction should process the module.

        Parameters
        ----------
        module
            Module metadata describing the candidate file.

        Returns
        -------
        bool
            True when the module maps to a Python source path.
        """
        return module.rel_path.endswith(".py")

    def process_module(self, module: ModuleRecord) -> Iterable[ModuleResult]:
        """
        Parse a module and emit CST rows.

        Parameters
        ----------
        module
            Module metadata describing the file to analyze.

        Returns
        -------
        Iterable[ModuleResult]
            Extraction result (single element list).
        """
        return [_process_module(module)]

    def persist_rows(self, gateway: StorageGateway, rows: Sequence[ModuleResult]) -> int:
        """
        Insert serialized CST rows into DuckDB.

        Parameters
        ----------
        gateway
            Gateway whose connection receives batched inserts.
        rows
            Extraction results yielded from worker processes.

        Returns
        -------
        int
            Number of CST rows persisted.
        """
        batch: list[Row] = []
        total = 0

        for result in rows:
            if result.error is not None:
                log.warning("Failed to parse %s: %s", result.rel_path, result.error)
            if result.rows:
                batch.extend(result.rows)
            if len(batch) >= FLUSH_EVERY:
                total += _flush_batch(gateway, batch)
                batch = []

        if batch:
            total += _flush_batch(gateway, batch)

        return total

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        Remove CST rows for modules scheduled for deletion.

        Parameters
        ----------
        gateway
            Gateway whose connection executes DELETE statements.
        rel_paths
            Relative module paths to remove from CST tables.
        """
        if not rel_paths:
            return
        gateway.con.execute(
            """
            DELETE FROM core.cst_nodes
            WHERE path IN (
                SELECT path FROM core.modules
                WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            )
            """,
            [self._repo, self._commit, list(rel_paths)],
        )


# Type assertion that CstPipeline implements IngestPipeline
_: type[IngestPipeline[ModuleResult]] = CstPipeline


def ingest_cst(
    gateway: StorageGateway,
    modules: Sequence[ModuleRecord],
    *,
    repo: str,
    commit: str,
    tracker: ChangeTracker | None = None,
    max_workers: int | None = None,
    executor_kind: str | None = None,
) -> PipelineResult:
    """
    Parse modules using LibCST and populate cst_nodes table.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    modules
        Available modules for processing.
    repo
        Repository identifier.
    commit
        Commit identifier.
    tracker
        Optional change tracker for incremental mode.
    max_workers
        Optional worker count override.
    executor_kind
        Optional override for executor selection ("thread" or "process").

    Returns
    -------
    PipelineResult
        Execution result with counts and timing.
    """
    workers = resolve_worker_count(
        CST_WORKER_CONFIG.env_var,
        explicit_count=max_workers,
        default_max=CST_WORKER_CONFIG.default_max,
    )

    pipeline = CstPipeline(repo=repo, commit=commit)
    config = PipelineConfig(
        worker_config=CST_WORKER_CONFIG,
        max_workers=workers,
        executor_kind=executor_kind,
    )

    return execute_pipeline(
        pipeline,
        gateway,
        modules,
        tracker=tracker,
        config=config,
    )


# Backward compatibility: keep old function signature
def ingest_cst_legacy(
    tracker: ChangeTracker,
    *,
    max_workers: int | None = None,
    executor_kind: str | None = None,
) -> None:
    """
    Legacy entry point for CST ingestion.

    Deprecated: Use ingest_cst() with explicit parameters instead.
    """
    from codeintel.ingestion.common import iter_modules
    from codeintel.storage.module_index import load_module_map

    module_map = load_module_map(
        tracker.gateway,
        tracker.change_request.repo,
        tracker.change_request.commit,
        language="python",
        logger=log,
    )

    modules = list(
        iter_modules(
            module_map,
            tracker.change_request.repo_root,
            logger=log,
            scan_profile=tracker.change_request.scan_profile,
        )
    )

    ingest_cst(
        tracker.gateway,
        modules,
        repo=tracker.change_request.repo,
        commit=tracker.change_request.commit,
        tracker=tracker,
        max_workers=max_workers,
        executor_kind=executor_kind,
    )
