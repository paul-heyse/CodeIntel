"""Extract LibCST concrete syntax trees into DuckDB tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import libcst as cst
from libcst import metadata

from codeintel.config.schemas.sql_builder import ensure_schema, prepared_statements
from codeintel.ingestion.common import (
    PROGRESS_LOG_EVERY,
    PROGRESS_LOG_INTERVAL,
    iter_modules,
    load_module_map,
    read_module_source,
)

log = logging.getLogger(__name__)
ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)


@dataclass
class CstRow:
    """Flattened CST node record."""

    path: str
    node_id: str
    kind: str
    span: dict[str, list[int]]
    text_preview: str
    parents: list[str]
    qnames: list[str]


class CstVisitor(cst.CSTVisitor):
    """Collect CST rows and lightweight scope info."""

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, rel_path: str, module_name: str, source: str) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source_lines = source.splitlines(keepends=True)

        self.cst_rows: list[CstRow] = []
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
        self._parent_kinds.append(type(node).__name__)
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

        self._record_cst_row(node)
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Pop scope/parent stacks on exit."""
        if isinstance(
            original_node,
            (
                cst.ClassDef,
                ASYNC_FUNC_DEF,
                cst.FunctionDef,
            ),
        ) and self._scope_stack:
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

    def _record_cst_row(self, node: cst.CSTNode) -> None:
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
        if 0 < start.line <= len(self.source_lines) and 0 < end.line <= len(self.source_lines):
            if start.line == end.line:
                line = self.source_lines[start.line - 1]
                snippet = line[start.column : end.column]
            else:
                lines = list(self.source_lines[start.line - 1 : end.line])
                if lines:
                    lines[0] = lines[0][start.column :]
                    lines[-1] = lines[-1][: end.column]
                    snippet = "".join(lines)

        kind = type(node).__name__
        parents = list(self._parent_kinds[:-1])
        qnames = [self._current_qualname()]
        node_id = f"{self.rel_path}:{kind}:{start.line}:{start.column}:{end.line}:{end.column}"

        self.cst_rows.append(
            CstRow(
                path=self.rel_path,
                node_id=node_id,
                kind=kind,
                span=span,
                text_preview=snippet[:200],
                parents=parents,
                qnames=qnames,
            )
        )


def _load_module_map(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> dict[str, str]:
    return load_module_map(con, repo, commit, logger=log)


def ingest_cst(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
) -> None:
    """Parse modules listed in core.modules using LibCST and populate cst_nodes."""
    repo_root = repo_root.resolve()
    module_map = _load_module_map(con, repo, commit)
    if not module_map:
        log.warning("No modules found in core.modules for %s@%s", repo, commit)
        return
    total_modules = len(module_map)
    log.info("Parsing CST for %d modules in %s@%s", total_modules, repo, commit)

    ensure_schema(con, "core.cst_nodes")
    cst_stmt = prepared_statements("core.cst_nodes")
    if cst_stmt.delete_sql is not None:
        con.execute(cst_stmt.delete_sql, [repo, commit])

    cst_values: list[list[object]] = []

    for record in iter_modules(
        module_map,
        repo_root,
        logger=log,
        log_every=PROGRESS_LOG_EVERY,
        log_interval=PROGRESS_LOG_INTERVAL,
    ):
        source = read_module_source(record, logger=log)
        if source is None:
            continue
        try:
            wrapper = metadata.MetadataWrapper(cst.parse_module(source))
            visitor = CstVisitor(rel_path=record.rel_path, module_name=record.module_name, source=source)
            wrapper.visit(visitor)
        except Exception:
            log.exception("Failed to parse %s", record.file_path)
            continue

        cst_values.extend(
            [
                [
                    row.path,
                    row.node_id,
                    row.kind,
                    row.span,
                    row.text_preview,
                    row.parents,
                    row.qnames,
                ]
                for row in visitor.cst_rows
            ]
        )

    if cst_values:
        con.executemany(cst_stmt.insert_sql, cst_values)

    log.info("CST extraction complete for %s@%s (%d modules)", repo, commit, total_modules)
