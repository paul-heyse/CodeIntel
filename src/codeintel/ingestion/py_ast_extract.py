"""Extract Python stdlib AST into DuckDB tables."""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb

from codeintel.config.models import PyAstIngestConfig
from codeintel.config.schemas.sql_builder import ensure_schema, prepared_statements
from codeintel.ingestion.common import (
    PROGRESS_LOG_EVERY,
    PROGRESS_LOG_INTERVAL,
    ModuleRecord,
    iter_modules,
    load_module_map,
    read_module_source,
)

log = logging.getLogger(__name__)


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

        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.AsyncFor)):
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and self._scope_stack:
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
        self._record_ast_row(
            node=node,
            info=AstRow(
                node_type="ClassDef",
                name=name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
                decorators=[self._decorator_to_str(d) for d in node.decorator_list],
                docstring=ast.get_docstring(node),
            ),
        )

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool) -> None:
        """Record function or async function definitions and push scope."""
        name = node.name
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{name}" if parent_qual else f"{self.module_name}.{name}"
        self._scope_stack.append(name)
        self.metrics.function_count += 1
        node_type = "AsyncFunctionDef" if is_async else "FunctionDef"
        self._record_ast_row(
            node=node,
            info=AstRow(
                node_type=node_type,
                name=name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
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


def ingest_python_ast(
    con: duckdb.DuckDBPyConnection,
    cfg: PyAstIngestConfig,
) -> None:
    """
    Parse modules listed in core.modules using the stdlib ast and populate tables.

    Parameters
    ----------
    con:
        Live DuckDB connection.
    cfg:
        Repository context (root, repo slug, commit).
    """
    repo_root = cfg.repo_root
    module_map = load_module_map(con, cfg.repo, cfg.commit, logger=log)
    if not module_map:
        return
    total_modules = len(module_map)
    log.info("Parsing Python AST for %d modules in %s@%s", total_modules, cfg.repo, cfg.commit)

    ensure_schema(con, "core.ast_nodes")
    ensure_schema(con, "core.ast_metrics")
    ast_nodes_stmt = prepared_statements("core.ast_nodes")
    ast_metrics_stmt = prepared_statements("core.ast_metrics")
    if ast_nodes_stmt.delete_sql is not None:
        con.execute(ast_nodes_stmt.delete_sql, [cfg.repo, cfg.commit])
    if ast_metrics_stmt.delete_sql is not None:
        con.execute(ast_metrics_stmt.delete_sql, [cfg.repo, cfg.commit])

    now = datetime.now(UTC)
    ast_values: list[list[object]] = []
    metric_values: list[list[object]] = []

    for record in iter_modules(
        module_map,
        repo_root,
        logger=log,
        log_every=PROGRESS_LOG_EVERY,
        log_interval=PROGRESS_LOG_INTERVAL,
    ):
        module_data = _collect_module_ast(record)
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
                now,
            ]
        )

    if ast_values:
        con.executemany(ast_nodes_stmt.insert_sql, ast_values)
    if metric_values:
        con.executemany(ast_metrics_stmt.insert_sql, metric_values)

    log.info(
        "AST extraction complete for %s@%s (%d modules)",
        cfg.repo,
        cfg.commit,
        total_modules,
    )
