"""Extract Python stdlib AST into DuckDB tables."""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

from codeintel.config.models import PyAstIngestConfig
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
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
DEFAULT_MAX_WORKERS = 16


@dataclass
class AstMetrics:
    """Aggregated metrics for a single file."""

    rel_path: str
    node_count: int = 0
    function_count: int = 0
    class_count: int = 0
    depths: list[int] = field(default_factory=list)
    complexity: float = 0.0

    @property
    def avg_depth(self) -> float:
        """Average nesting depth for the file."""
        return (sum(self.depths) / len(self.depths)) if self.depths else 0.0

    @property
    def max_depth(self) -> int:
        """Maximum nesting depth for the file."""
        return max(self.depths) if self.depths else 0


@dataclass
class AstRow:
    """Flattened AST row payload."""

    node_type: str
    name: str | None
    qualname: str | None
    parent_qualname: str | None
    decorator_start_line: int | None
    decorator_end_line: int | None
    decorators: list[str]
    docstring: str | None


class AstVisitor(ast.NodeVisitor):
    """Collect Python AST nodes and file metrics."""

    def __init__(self, rel_path: str, module_name: str) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.ast_rows: list[list[object]] = []
        self.metrics = AstMetrics(rel_path=rel_path)
        self._scope_stack: list[str] = []
        self._depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        """Track complexity, scope, and depth while visiting."""
        self.metrics.node_count += 1
        self._depth += 1
        self.metrics.depths.append(self._depth)

        if isinstance(
            node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.AsyncFor)
        ):
            self.metrics.complexity += 1
        if isinstance(node, ast.FunctionDef):
            self._record_function(node, is_async=False)
        elif isinstance(node, ast.AsyncFunctionDef):
            self._record_function(node, is_async=True)
        elif isinstance(node, ast.ClassDef):
            self._record_class(node)
        elif isinstance(node, ast.Module):
            self._record_module(node)

        super().generic_visit(node)
        self._depth -= 1
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and self._scope_stack
        ):
            self._scope_stack.pop()

    def _current_qualname(self) -> str:
        """
        Return current module-qualified scope.

        Returns
        -------
        str
            Fully qualified name for the active scope.
        """
        if not self._scope_stack:
            return self.module_name
        return f"{self.module_name}." + ".".join(self._scope_stack)

    def _record_module(self, node: ast.Module) -> None:
        """Record the module node and reset scope."""
        qualname = self.module_name
        self._scope_stack = []
        self._record_ast_row(
            node=node,
            info=AstRow(
                node_type="Module",
                name=self.module_name.split(".")[-1],
                qualname=qualname,
                parent_qualname=None,
                decorator_start_line=None,
                decorator_end_line=None,
                decorators=[],
                docstring=ast.get_docstring(node),
            ),
        )

    def _record_class(self, node: ast.ClassDef) -> None:
        """Record a class definition and push scope."""
        name = node.name
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{name}" if parent_qual else f"{self.module_name}.{name}"
        self._scope_stack.append(name)
        self.metrics.class_count += 1
        dec_start, dec_end = self._decorator_span(node.decorator_list)
        self._record_ast_row(
            node=node,
            info=AstRow(
                node_type="ClassDef",
                name=name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
                decorator_start_line=dec_start,
                decorator_end_line=dec_end,
                decorators=[self._decorator_to_str(d) for d in node.decorator_list],
                docstring=ast.get_docstring(node),
            ),
        )

    def _record_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        """Record function or async function definitions and push scope."""
        name = node.name
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{name}" if parent_qual else f"{self.module_name}.{name}"
        self._scope_stack.append(name)
        self.metrics.function_count += 1
        node_type = "AsyncFunctionDef" if is_async else "FunctionDef"
        dec_start, dec_end = self._decorator_span(node.decorator_list)
        self._record_ast_row(
            node=node,
            info=AstRow(
                node_type=node_type,
                name=name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
                decorator_start_line=dec_start,
                decorator_end_line=dec_end,
                decorators=[self._decorator_to_str(d) for d in node.decorator_list],
                docstring=ast.get_docstring(node),
            ),
        )

    @staticmethod
    def _decorator_to_str(node: ast.AST) -> str:
        """
        Render a decorator expression safely.

        Returns
        -------
        str
            Source-like representation or fallback type name.
        """
        try:
            return ast.unparse(node)
        except (AttributeError, SyntaxError, TypeError, ValueError):
            return type(node).__name__

    @staticmethod
    def _decorator_span(
        decorators: Sequence[ast.AST],
    ) -> tuple[int | None, int | None]:
        """
        Determine the span covered by decorators, if present.

        Returns
        -------
        tuple[int | None, int | None]
            Minimum and maximum decorator lines (inclusive).
        """
        if not decorators:
            return None, None
        start: int | None = None
        end: int | None = None
        for dec in decorators:
            dec_start = getattr(dec, "lineno", None)
            dec_end = getattr(dec, "end_lineno", None) or dec_start
            if dec_start is not None:
                start = dec_start if start is None else min(start, dec_start)
            if dec_end is not None:
                end = dec_end if end is None else max(end, dec_end)
        return start, end

    def _record_ast_row(
        self,
        node: ast.AST,
        info: AstRow,
    ) -> None:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        h = hashlib.blake2b(
            f"{self.rel_path}:{info.node_type}:{info.qualname}:{lineno}:{end_lineno}".encode(),
            digest_size=16,
        ).hexdigest()
        self.ast_rows.append(
            [
                self.rel_path,
                info.node_type,
                info.name,
                info.qualname,
                lineno,
                end_lineno,
                info.decorator_start_line,
                info.decorator_end_line,
                col,
                end_col,
                info.parent_qualname,
                info.decorators,
                info.docstring,
                h,
            ]
        )


def _collect_module_ast(record: ModuleRecord) -> tuple[list[list[object]], AstMetrics] | None:
    """
    Parse a module from disk and return serialized AST rows and metrics.

    Returns
    -------
    tuple[list[list[object]], AstMetrics] | None
        Serialized rows plus metrics, or None if parsing fails.
    """
    source = read_module_source(record, logger=log)
    if source is None:
        return None

    try:
        tree = ast.parse(source, filename=str(record.file_path))
    except (SyntaxError, ValueError) as exc:
        log.warning("Failed to parse %s: %s", record.file_path, exc)
        return None

    visitor = AstVisitor(rel_path=record.rel_path, module_name=record.module_name)
    try:
        visitor.visit(tree)
    except (RecursionError, ValueError) as exc:
        log.warning("AST visit failed for %s: %s", record.file_path, exc)
        return None

    return visitor.ast_rows, visitor.metrics


def _resolve_worker_count() -> int:
    """
    Resolve AST worker pool size.

    Returns
    -------
    int
        Worker count derived from environment or CPU.
    """
    env_workers = os.getenv("CODEINTEL_AST_WORKERS")
    if env_workers:
        try:
            value = int(env_workers)
            if value > 0:
                return value
        except ValueError:
            log.warning("Ignoring invalid CODEINTEL_AST_WORKERS=%s", env_workers)

    cpu_count = os.cpu_count() or 1
    return min(DEFAULT_MAX_WORKERS, max(2, cpu_count // 2))


def _delete_existing_ast_rows(
    gateway: StorageGateway, cfg: PyAstIngestConfig, rel_paths: list[str]
) -> None:
    """Remove stale AST rows prior to incremental inserts."""
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM core.ast_nodes
            WHERE path IN (
                SELECT path FROM core.modules WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            )
            """,
            [cfg.repo, cfg.commit, rel_paths],
        )
        gateway.con.execute(
            """
            DELETE FROM core.ast_metrics
            WHERE rel_path IN (
                SELECT path FROM core.modules WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            )
            """,
            [cfg.repo, cfg.commit, rel_paths],
        )
        return
    run_batch(
        gateway,
        "core.ast_nodes",
        [],
        delete_params=[cfg.repo, cfg.commit],
    )
    run_batch(
        gateway,
        "core.ast_metrics",
        [],
        delete_params=[cfg.repo, cfg.commit],
    )


def ingest_python_ast(
    gateway: StorageGateway,
    cfg: PyAstIngestConfig,
    scan_config: ScanConfig | None = None,
    change_set: ChangeSet | None = None,
) -> None:
    """
    Parse modules listed in core.modules using the stdlib ast and populate tables.

    Parameters
    ----------
    gateway:
        StorageGateway providing access to the DuckDB database.
    cfg:
        Repository context (root, repo slug, commit).
    scan_config:
        Optional scan configuration controlling iteration logging cadence.
    change_set:
        Optional precomputed change set enabling incremental ingestion.

    """
    repo_root = cfg.repo_root.resolve()
    module_map = load_module_map(gateway, cfg.repo, cfg.commit, language="python", logger=log)
    if should_skip_empty(module_map, logger=log):
        return
    active_change_set = change_set or compute_changes(
        gateway,
        ChangeRequest(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=repo_root,
            scan_config=scan_config,
            logger=log,
        ),
    )
    to_reparse = list(active_change_set.added + active_change_set.modified)
    if not to_reparse and not active_change_set.deleted:
        log.info("No AST changes detected for %s@%s", cfg.repo, cfg.commit)
        return
    total_modules = (
        len(to_reparse) if to_reparse else (0 if active_change_set.deleted else len(module_map))
    )
    log.info("Parsing Python AST for %d modules in %s@%s", total_modules, cfg.repo, cfg.commit)
    start_ts = time.perf_counter()

    ast_values: list[list[object]] = []
    metric_values: list[list[object]] = []

    records: Sequence[ModuleRecord]
    if to_reparse:
        records = to_reparse
    else:
        records = tuple(
            iter_modules(
                module_map,
                repo_root,
                logger=log,
                scan_config=scan_config,
            )
        )

    _delete_existing_ast_rows(
        gateway,
        cfg,
        [record.rel_path for record in to_reparse + active_change_set.deleted],
    )

    if records:
        with ProcessPoolExecutor(max_workers=_resolve_worker_count()) as pool:
            for module_data in pool.map(_collect_module_ast, records):
                if module_data is None:
                    continue
                ast_rows, metrics = module_data
                ast_values.extend(ast_rows)
                metric_values.append(
                    [
                        metrics.rel_path,
                        metrics.node_count,
                        metrics.function_count,
                        metrics.class_count,
                        metrics.avg_depth,
                        metrics.max_depth,
                        metrics.complexity,
                        datetime.now(UTC),
                    ]
                )

    run_batch(
        gateway,
        "core.ast_nodes",
        ast_values,
        delete_params=None,
    )
    run_batch(
        gateway,
        "core.ast_metrics",
        metric_values,
        delete_params=None,
    )

    duration = time.perf_counter() - start_ts
    log.info(
        "AST extraction complete for %s@%s (%d modules, %d rows, %.2fs)",
        cfg.repo,
        cfg.commit,
        total_modules,
        len(ast_values),
        duration,
    )
