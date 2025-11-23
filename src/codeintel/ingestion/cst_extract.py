"""Extract LibCST concrete syntax trees into DuckDB tables."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import duckdb
import libcst as cst
from libcst import metadata

from codeintel.ingestion.common import (
    PROGRESS_LOG_EVERY,
    PROGRESS_LOG_INTERVAL,
    ModuleRecord,
    iter_modules,
    load_module_map,
    read_module_source,
    run_batch,
    should_skip_empty,
)

log = logging.getLogger(__name__)
ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)
FLUSH_EVERY = 10_000
DEFAULT_MAX_WORKERS = 16
Row = tuple[str, str, str, dict[str, list[int]], str, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class ModuleResult:
    """Rows and errors returned from processing a single module."""

    rel_path: str
    rows: list[Row]
    error: str | None = None


class CstVisitor(cst.CSTVisitor):
    """Collect CST rows and lightweight scope info."""

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, rel_path: str, module_name: str, source: str) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source = source
        self._line_offsets: list[int] = []
        offset = 0
        for line in source.splitlines(keepends=True):
            self._line_offsets.append(offset)
            offset += len(line)

        self.cst_rows: list[Row] = []
        self._seen_ids: set[str] = set()
        self._scope_stack: list[str] = []
        self._parent_kinds: list[str] = []

    def on_visit(self, node: cst.CSTNode) -> bool:
        """
        Handle pre-visit bookkeeping and record CST row.

        Returns
        -------
        bool
            Always True to continue traversal.
        """
        kind = type(node).__name__
        self._parent_kinds.append(kind)
        if isinstance(
            node,
            (
                cst.ClassDef,
                cst.FunctionDef,
                ASYNC_FUNC_DEF,
            ),
        ):
            name_node = getattr(node, "name", None)
            if name_node is not None and hasattr(name_node, "value"):
                self._scope_stack.append(name_node.value)

        self._record_cst_row(node, kind)
        return True

    @staticmethod
    def _should_capture(node: cst.CSTNode) -> bool:
        """
        Filter nodes to capture only high-value structural elements.

        Parameters
        ----------
        node : cst.CSTNode
            Candidate CST node to evaluate.

        Returns
        -------
        bool
            True if the node should be recorded for ingestion.
        """
        return isinstance(
            node,
            (
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
        )

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Pop scope/parent stacks on exit."""
        if (
            isinstance(
                original_node,
                (
                    cst.ClassDef,
                    ASYNC_FUNC_DEF,
                    cst.FunctionDef,
                ),
            )
            and self._scope_stack
        ):
            self._scope_stack.pop()
        self._parent_kinds.pop()

    def _current_qualname(self) -> str:
        """
        Return current qualified name based on the scope stack.

        Returns
        -------
        str
            Module-prefixed qualname of the current scope.
        """
        if not self._scope_stack:
            return self.module_name
        return f"{self.module_name}." + ".".join(self._scope_stack)

    def _record_cst_row(self, node: cst.CSTNode, kind: str) -> None:
        if not self._should_capture(node):
            return

        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return
        if not isinstance(pos, metadata.CodeRange):
            return

        start = pos.start
        end = pos.end
        span = {"start": [start.line, start.column], "end": [end.line, end.column]}

        snippet = ""
        try:
            start_idx = self._line_offsets[start.line - 1] + start.column
            end_idx = self._line_offsets[end.line - 1] + end.column
            snippet = self.source[start_idx:end_idx]
        except (IndexError, ValueError):
            snippet = ""

        parents = tuple(self._parent_kinds[:-1])
        qnames = (self._current_qualname(),)
        node_id = f"{self.rel_path}:{kind}:{start.line}:{start.column}:{end.line}:{end.column}"

        if node_id in self._seen_ids:
            return
        self._seen_ids.add(node_id)

        self.cst_rows.append(
            (
                self.rel_path,
                node_id,
                kind,
                span,
                snippet[:200],
                parents,
                qnames,
            )
        )


def _load_module_map(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> dict[str, str]:
    return load_module_map(con, repo, commit, logger=log)


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
) -> None:
    """Parse modules listed in core.modules using LibCST and populate cst_nodes."""
    repo_root = repo_root.resolve()
    module_map = _load_module_map(con, repo, commit)
    if should_skip_empty(module_map, logger=log):
        return
    total_modules = len(module_map)
    log.info("Parsing CST for %d modules in %s@%s", total_modules, repo, commit)

    cst_values: list[Row] = []
    records: list[ModuleRecord] = []

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
            log_every=PROGRESS_LOG_EVERY,
            log_interval=PROGRESS_LOG_INTERVAL,
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
    log.info("CST extraction complete for %s@%s (%d modules)", repo, commit, total_modules)


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

    return ModuleResult(rel_path=record.rel_path, rows=visitor.cst_rows, error=None)
