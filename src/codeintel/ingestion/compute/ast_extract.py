"""AST extraction step with port injection.

This module provides a pure domain logic implementation for extracting
Python AST nodes and metrics, using ports for all I/O operations.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.compute.base import BaseExtractStep, ExecutionResult
from codeintel.ingestion.row_serialization import row_serializer_for_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)
AST_NODES_TABLE_KEY = "core.ast_nodes"
AST_METRICS_TABLE_KEY = "core.ast_metrics"


@dataclass
class AstMetrics:
    """Aggregated metrics for a single file.

    Attributes
    ----------
    rel_path
        Relative path to the file.
    node_count
        Total number of AST nodes.
    function_count
        Number of function definitions.
    class_count
        Number of class definitions.
    depths
        List of nesting depths.
    complexity
        Cyclomatic complexity estimate.
    """

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
class AstRowInfo:
    """Flattened AST row payload."""

    node_type: str
    name: str | None
    qualname: str | None
    parent_qualname: str | None
    decorator_start_line: int | None
    decorator_end_line: int | None
    decorators: list[str]
    docstring: str | None


@dataclass
class ModuleAstResult:
    """Result from processing a single module's AST."""

    ast_rows: list[dict[str, object]]
    metric_row: dict[str, object] | None


class AstVisitor(ast.NodeVisitor):
    """Collect Python AST nodes and file metrics."""

    def __init__(self, rel_path: str, module_name: str) -> None:
        """Initialize visitor.

        Parameters
        ----------
        rel_path
            Relative path to the file.
        module_name
            Python module name.
        """
        self.rel_path = rel_path
        self.module_name = module_name
        self.ast_rows: list[dict[str, object]] = []
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
        """Return current module-qualified scope.

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
            info=AstRowInfo(
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
            info=AstRowInfo(
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
            info=AstRowInfo(
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
        """Render a decorator expression safely.

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
        """Determine the span covered by decorators, if present.

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
        info: AstRowInfo,
    ) -> None:
        """Record an AST row for storage."""
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        h = hashlib.blake2b(
            f"{self.rel_path}:{info.node_type}:{info.qualname}:{lineno}:{end_lineno}".encode(),
            digest_size=16,
        ).hexdigest()
        self.ast_rows.append(
            {
                "path": self.rel_path,
                "node_type": info.node_type,
                "name": info.name,
                "qualname": info.qualname,
                "lineno": lineno,
                "end_lineno": end_lineno,
                "decorator_start_line": info.decorator_start_line,
                "decorator_end_line": info.decorator_end_line,
                "col_offset": col,
                "end_col_offset": end_col,
                "parent_qualname": info.parent_qualname,
                "decorators": info.decorators,
                "docstring": info.docstring,
                "hash": h,
            }
        )


def _build_metric_row(metrics: AstMetrics) -> dict[str, object]:
    """Build a metric row from AstMetrics.

    Returns
    -------
    list[object]
        Row data for metric table.
    """
    return {
        "rel_path": metrics.rel_path,
        "node_count": metrics.node_count,
        "function_count": metrics.function_count,
        "class_count": metrics.class_count,
        "avg_depth": metrics.avg_depth,
        "max_depth": metrics.max_depth,
        "complexity": metrics.complexity,
        "generated_at": datetime.now(UTC),
    }


def _extract_module_ast(
    module: ModuleRecord,
    source: str,
) -> ModuleAstResult | None:
    """Extract AST from module source.

    Parameters
    ----------
    module
        Module record with metadata.
    source
        Module source code.

    Returns
    -------
    ModuleAstResult | None
        Extraction result, or None if parsing fails.
    """
    try:
        tree = ast.parse(source, filename=str(module.file_path))
    except (SyntaxError, ValueError) as exc:
        log.warning("Failed to parse %s: %s", module.file_path, exc)
        return None

    visitor = AstVisitor(rel_path=module.rel_path, module_name=module.module_name)
    try:
        visitor.visit(tree)
    except (RecursionError, ValueError) as exc:
        log.warning("AST visit failed for %s: %s", module.file_path, exc)
        return None

    return ModuleAstResult(
        ast_rows=visitor.ast_rows,
        metric_row=_build_metric_row(visitor.metrics),
    )


class AstExtractStep(BaseExtractStep):
    """AST extraction step with port injection.

    This step extracts Python AST nodes and metrics from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for reading module source.
    """

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> ExecutionResult:
        """Execute AST extraction on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        ExecutionResult
            Execution result with row counts.
        """
        ast_rows: list[tuple[object, ...]] = []
        metric_rows: list[tuple[object, ...]] = []
        warnings: list[str] = []
        ast_serializer = row_serializer_for_table_key(AST_NODES_TABLE_KEY)
        metrics_serializer = row_serializer_for_table_key(AST_METRICS_TABLE_KEY)

        for module, source in self._iter_python_sources(modules):
            result = _extract_module_ast(module, source)
            if result is None:
                warnings.append(f"Failed to extract AST from {module.rel_path}")
                continue

            ast_rows.extend(ast_serializer(row) for row in result.ast_rows)
            if result.metric_row is not None:
                metric_rows.append(metrics_serializer(result.metric_row))

        table_counts = self._write_and_count(
            AST_NODES_TABLE_KEY,
            ast_rows,
            repo=repo,
            commit=commit,
        )
        if metric_rows:
            result = self._storage.write_batch(AST_METRICS_TABLE_KEY, metric_rows)
            table_counts[AST_METRICS_TABLE_KEY] = result.rows_affected

        log.info(
            "AST extraction: repo=%s commit=%s ast_rows=%d metrics=%d",
            repo,
            commit,
            len(ast_rows),
            len(metric_rows),
        )

        return ExecutionResult.ok(
            table_counts=table_counts,
            warnings=tuple(warnings),
        )


__all__ = [
    "AstExtractStep",
    "AstMetrics",
    "AstRowInfo",
    "AstVisitor",
    "ModuleAstResult",
]
