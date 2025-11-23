"""Extract LibCST concrete syntax trees into DuckDB tables."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import duckdb
import libcst as cst
from libcst import metadata

from codeintel.ingestion.common import (
    ModuleRecord,
    iter_modules,
    load_module_map,
    read_module_source,
    run_batch,
    should_skip_empty,
)
from codeintel.ingestion.cst_utils import CstCaptureConfig, CstCaptureVisitor
from codeintel.ingestion.source_scanner import ScanConfig

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
    con: duckdb.DuckDBPyConnection,
    rows: list[Row],
) -> list[Row]:
    if not rows:
        return []
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
        con,
        "core.cst_nodes",
        normalized_rows,
        delete_params=None,
    )
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
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
    scan_config: ScanConfig | None = None,
) -> None:
    """
    Parse modules listed in core.modules using LibCST and populate cst_nodes.

    Parameters
    ----------
    con:
        DuckDB connection.
    repo_root:
        Repository root containing source files.
    repo:
        Repository slug.
    commit:
        Commit hash for the current run.
    scan_config:
        Optional scan configuration to tune logging cadence.
    """
    repo_root = repo_root.resolve()
    module_map = load_module_map(con, repo, commit, language="python", logger=log)
    if should_skip_empty(module_map, logger=log):
        return
    total_modules = len(module_map)
    log.info("Parsing CST for %d modules in %s@%s", total_modules, repo, commit)

    cst_values: list[Row] = []
    records: list[ModuleRecord] = []
    start_ts = time.perf_counter()

    # Clear existing data first if we are starting a new ingestion
    # We can do this by running an empty batch with delete_params, or just relying on the first flush.
    # Relying on first flush is safer if we yield at least one row, but if no rows are found,
    # we still want to clear old data.
    run_batch(con, "core.cst_nodes", [], delete_params=[repo, commit])

    records = list(
        iter_modules(
            module_map,
            repo_root,
            logger=log,
            scan_config=scan_config,
        )
    )

    worker_count = _resolve_worker_count()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for result in pool.map(_process_module, records):
            if result.error is not None:
                log.warning("Failed to parse %s: %s", result.rel_path, result.error)
            if result.rows:
                cst_values.extend(result.rows)
            if len(cst_values) >= FLUSH_EVERY:
                cst_values = _flush_batch(con, cst_values)

    _flush_batch(con, cst_values)
    duration = time.perf_counter() - start_ts
    log.info(
        "CST extraction complete for %s@%s (%d modules, %d rows, %.2fs)",
        repo,
        commit,
        total_modules,
        len(cst_values),
        duration,
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
