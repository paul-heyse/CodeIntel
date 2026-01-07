"""AST extraction step with port injection.

This module provides a pure domain logic implementation for extracting
Python AST nodes and metrics, using ports for all I/O operations.
"""

from __future__ import annotations

import ast
import io
import logging
import tokenize
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import AstExtractOptions
from codeintel.core.columnar.rows import (
    ColumnarBatchCollector,
    ColumnarRows,
    columnar_batch_collector_for_table_key,
    empty_table_for_table,
)
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.context import IngestionContext, resolve_repo_commit
from codeintel.ingestion.infrastructure.ast_facts import (
    AstCollectContext,
    AstNodeRecord,
    ast_node_id,
    collect_ast_nodes,
)
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pyarrow as pa

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

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


@dataclass(frozen=True)
class _DefInfo:
    """Definition metadata for qualname-bearing AST nodes."""

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


@dataclass(frozen=True)
class AstExtractResult:
    """Result bundle for AST extraction."""

    result: ExecutionResult
    ast_rows: ColumnarRows = field(default_factory=dict)
    metric_rows: ColumnarRows = field(default_factory=dict)
    ast_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(AST_NODES_TABLE_KEY)
    )
    metric_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(AST_METRICS_TABLE_KEY)
    )
    ast_row_count: int = 0
    metric_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _AstCollectors:
    ast_nodes: ColumnarBatchCollector
    metrics: ColumnarBatchCollector


def _build_ast_collectors(options: AstExtractOptions) -> _AstCollectors:
    return _AstCollectors(
        ast_nodes=columnar_batch_collector_for_table_key(
            AST_NODES_TABLE_KEY,
            batch_size=options.batch_size,
            extras_policy="retain",
        ),
        metrics=columnar_batch_collector_for_table_key(
            AST_METRICS_TABLE_KEY,
            batch_size=options.batch_size,
            extras_policy="retain",
        ),
    )


def _flush_ast_collectors(collectors: _AstCollectors) -> None:
    collectors.ast_nodes.flush()
    collectors.metrics.flush()


class AstVisitor(ast.NodeVisitor):
    """Collect AST metrics and definition metadata."""

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
        self.metrics = AstMetrics(rel_path=rel_path)
        self.def_info_by_node_id: dict[int, _DefInfo] = {}
        self.ast_rows: list[dict[str, object]] = []
        self._scope_stack: list[str] = []
        self._depth = 0
        self._root: ast.AST | None = None

    def visit(self, node: ast.AST) -> None:
        """Record the root node before delegating to the base visitor."""
        if self._root is None:
            self._root = node
        super().visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Track complexity, scope, and depth while visiting."""
        self.metrics.node_count += 1
        self._depth += 1
        self.metrics.depths.append(self._depth)

        if isinstance(
            node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.AsyncFor)
        ):
            self.metrics.complexity += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._record_function(node)
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

    def build_ast_rows(self, source_text: str) -> list[dict[str, object]]:
        """Build AST rows from the collected visit metadata.

        Parameters
        ----------
        source_text
            Source text used for span and offset calculations.

        Returns
        -------
        list[dict[str, object]]
            AST row payloads for serialization.
        """
        if self._root is None:
            return []
        source_bytes = source_text.encode("utf-8", errors="replace")
        _, source_index = _build_source_index(source_bytes)
        records = collect_ast_nodes(
            source_text,
            source_index,
            node_id_factory=lambda node, span: ast_node_id(
                self.rel_path,
                type(node).__name__,
                span,
            ),
            context=AstCollectContext(
                source_label=self.rel_path,
                parsed=self._root,
            ),
        )
        self.ast_rows = _build_ast_rows(self.rel_path, records, self.def_info_by_node_id)
        return self.ast_rows

    def _record_module(self, node: ast.Module) -> None:
        """Record the module node and reset scope."""
        qualname = self.module_name
        self._scope_stack = []
        self._store_def_info(
            node,
            _DefInfo(
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
        self._store_def_info(
            node,
            _DefInfo(
                name=name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
                decorator_start_line=dec_start,
                decorator_end_line=dec_end,
                decorators=[self._decorator_to_str(d) for d in node.decorator_list],
                docstring=ast.get_docstring(node),
            ),
        )

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Record function or async function definitions and push scope."""
        name = node.name
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{name}" if parent_qual else f"{self.module_name}.{name}"
        self._scope_stack.append(name)
        self.metrics.function_count += 1
        dec_start, dec_end = self._decorator_span(node.decorator_list)
        self._store_def_info(
            node,
            _DefInfo(
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
    def _normalize_line(value: int | None) -> int | None:
        if not isinstance(value, int):
            return None
        return max(value - 1, 0)

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
            normalized_start = AstVisitor._normalize_line(dec_start)
            normalized_end = AstVisitor._normalize_line(dec_end)
            if normalized_start is not None:
                start = normalized_start if start is None else min(start, normalized_start)
            if normalized_end is not None:
                end = normalized_end if end is None else max(end, normalized_end)
        return start, end

    def _store_def_info(self, node: ast.AST, info: _DefInfo) -> None:
        self.def_info_by_node_id[id(node)] = info


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_type_ignores(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    entries = [item for item in value if isinstance(item, dict)]
    return entries or None


def _build_ast_rows(
    rel_path: str,
    records: list[AstNodeRecord],
    def_info: dict[int, _DefInfo],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        info = def_info.get(id(record.node))
        extras = record.extras or {}
        name = info.name if info is not None else _coerce_str(extras.get("name"))
        decorators = info.decorators if info is not None and info.decorators else None
        docstring = info.docstring if info is not None else None
        row: dict[str, object] = {
            "path": rel_path,
            "node_type": record.kind,
            "name": name,
            "qualname": info.qualname if info is not None else None,
            "lineno": record.span.start_line,
            "end_lineno": record.span.end_line,
            "decorator_start_line": info.decorator_start_line if info is not None else None,
            "decorator_end_line": info.decorator_end_line if info is not None else None,
            "col_offset": record.span.start_col_utf8,
            "end_col_offset": record.span.end_col_utf8,
            "start_byte": record.span.start_byte,
            "end_byte": record.span.end_byte,
            "parent_qualname": info.parent_qualname if info is not None else None,
            "decorators": decorators,
            "docstring": docstring,
            "ctx": _coerce_str(extras.get("ctx")),
            "type_comment": _coerce_str(extras.get("type_comment")),
            "type_ignores": _coerce_type_ignores(extras.get("type_ignores")),
            "identifier": _coerce_str(extras.get("identifier")),
            "attribute": _coerce_str(extras.get("attribute")),
            "imported": _coerce_str(extras.get("imported")),
            "asname": _coerce_str(extras.get("asname")),
            "module": _coerce_str(extras.get("module")),
            "level": _coerce_int(extras.get("level")),
            "constant_kind": _coerce_str(extras.get("constant_kind")),
            "hash": record.node_id,
        }
        rows.append(row)
    return rows


def _decode_source_bytes(source_bytes: bytes) -> tuple[str, str]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
    except SyntaxError:
        encoding = "utf-8"
    try:
        return source_bytes.decode(encoding), encoding
    except UnicodeDecodeError:
        return source_bytes.decode(encoding, errors="replace"), encoding


def _build_source_index(source_bytes: bytes) -> tuple[str, LineIndexedSource]:
    source_text, encoding = _decode_source_bytes(source_bytes)
    source_index = LineIndexedSource(source_text, source_bytes, encoding=encoding)
    return source_text, source_index


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
    source_text: str,
    source_index: LineIndexedSource,
    *,
    tree: ast.AST | None = None,
) -> ModuleAstResult | None:
    """Extract AST from module source.

    Parameters
    ----------
    module
        Module record with metadata.
    source_text
        Module source code text.
    source_index
        Line index used for span lookups.
    tree
        Pre-parsed AST when already available.

    Returns
    -------
    ModuleAstResult | None
        Extraction result, or None if parsing fails.
    """
    if tree is None:
        try:
            tree = ast.parse(source_text, filename=str(module.file_path), type_comments=True)
        except (SyntaxError, ValueError, TypeError) as exc:
            log.warning("Failed to parse %s: %s", module.file_path, exc)
            return None

    visitor = AstVisitor(rel_path=module.rel_path, module_name=module.module_name)
    try:
        visitor.visit(tree)
    except (RecursionError, ValueError, TypeError) as exc:
        log.warning("AST visit failed for %s: %s", module.file_path, exc)
        return None

    records = collect_ast_nodes(
        source_text,
        source_index,
        node_id_factory=lambda node, span: ast_node_id(
            module.rel_path,
            type(node).__name__,
            span,
        ),
        context=AstCollectContext(
            source_label=module.rel_path,
            parsed=tree,
        ),
    )
    return ModuleAstResult(
        ast_rows=_build_ast_rows(module.rel_path, records, visitor.def_info_by_node_id),
        metric_row=_build_metric_row(visitor.metrics),
    )


class AstExtractStep(BaseExtractStep):
    """AST extraction step with port injection.

    This step extracts Python AST nodes and metrics from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    """

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: AstExtractOptions | None = None,
        frontend: PyFrontend | None = None,
    ) -> None:
        super().__init__(discovery=discovery, frontend=frontend)
        self._options = options or AstExtractOptions()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        context: IngestionContext | None = None,
    ) -> AstExtractResult:
        """Execute AST extraction on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        context
            Optional ingestion context supplying repo/commit defaults.

        Returns
        -------
        AstExtractResult
            Result bundle with row tuples and execution status.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        options = self._options
        try:
            collectors = _build_ast_collectors(options)
        except (KeyError, RuntimeError) as exc:
            return AstExtractResult(result=ExecutionResult.failed(str(exc)))
        warnings: list[str] = []
        for module, source_text, source_index, tree in self._iter_python_source_bundles(modules):
            result = _extract_module_ast(module, source_text, source_index, tree=tree)
            if result is None:
                warnings.append(f"Failed to extract AST from {module.rel_path}")
                continue

            if result.ast_rows:
                collectors.ast_nodes.extend(result.ast_rows)
            if result.metric_row is not None:
                collectors.metrics.append(result.metric_row)
            _flush_ast_collectors(collectors)

        log.info(
            "AST extraction: repo=%s commit=%s ast_rows=%d metrics=%d",
            resolved_repo,
            resolved_commit,
            collectors.ast_nodes.row_count,
            collectors.metrics.row_count,
        )

        ast_rows_table = collectors.ast_nodes.to_table()
        metric_rows_table = collectors.metrics.to_table()
        return AstExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            ast_rows={},
            metric_rows={},
            ast_rows_reader=ast_rows_table,
            metric_rows_reader=metric_rows_table,
            ast_row_count=collectors.ast_nodes.row_count,
            metric_row_count=collectors.metrics.row_count,
        )

    def _iter_python_source_bundles(
        self,
        modules: Sequence[ModuleRecord],
    ) -> Iterable[tuple[ModuleRecord, str, LineIndexedSource, ast.AST | None]]:
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            if self._frontend is not None:
                bundle = self._frontend.get_source_bundle(module)
                if bundle is None:
                    continue
                tree = self._frontend.get_ast(module)
                yield module, bundle.source_text, bundle.source_index, tree
                continue
            source_bytes = self._discovery.read_module_bytes(module)
            if source_bytes is None:
                source_text = self._discovery.read_module_source(module)
                if source_text is None:
                    continue
                source_bytes = source_text.encode("utf-8", errors="replace")
            source_text, source_index = _build_source_index(source_bytes)
            yield module, source_text, source_index, None


__all__ = [
    "AstExtractResult",
    "AstExtractStep",
    "AstMetrics",
    "AstVisitor",
    "ModuleAstResult",
]
