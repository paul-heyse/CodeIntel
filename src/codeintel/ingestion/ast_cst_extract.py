"""Extract AST/CST structures and metrics into DuckDB tables."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import libcst as cst
from libcst import metadata

log = logging.getLogger(__name__)


@dataclass
class AstRow:
    """
    Flattened AST node record emitted for each module.

    Attributes mirror the `core.ast_nodes` schema, capturing node identity,
    location, decorators, and parent relationships.
    """

    path: str
    node_type: str
    name: str | None
    qualname: str | None
    lineno: int | None
    end_lineno: int | None
    col_offset: int | None
    end_col_offset: int | None
    parent_qualname: str | None
    decorators: list[str]
    docstring: str | None
    hash: str


@dataclass
class CstRow:
    """
    Flattened CST node record used to persist LibCST metadata.

    Captures node identity, span coordinates, text preview, parent kinds, and
    qualified names derived during traversal.
    """

    path: str
    node_id: str
    kind: str
    span: dict
    text_preview: str
    parents: list[str]
    qnames: list[str]


@dataclass
class FileMetrics:
    """
    Aggregated metrics describing a single Python file's structure.

    Includes counts of nodes, functions, and classes alongside depth and
    heuristic complexity information.
    """

    rel_path: str
    node_count: int = 0
    function_count: int = 0
    class_count: int = 0
    depths: list[int] = field(default_factory=list)
    complexity: float = 0.0  # heuristic

    @property
    def avg_depth(self) -> float:
        """Average syntax tree depth for visited nodes."""
        return (sum(self.depths) / len(self.depths)) if self.depths else 0.0

    @property
    def max_depth(self) -> int:
        """Maximum syntax tree depth encountered."""
        return max(self.depths) if self.depths else 0


class AstCstVisitor(cst.CSTVisitor):
    """
    Collect AST rows, CST rows, and structural metrics for a module traversal.

    The visitor increments complexity counters, tracks scope, and captures
    position metadata for every LibCST node it encounters.
    """

    METADATA_DEPENDENCIES = (metadata.PositionProvider, metadata.ParentNodeProvider)

    def __init__(self, rel_path: str, module_name: str, source: str) -> None:
        """Initialize visitor state for the target module."""
        self.rel_path = rel_path
        self.module_name = module_name
        self.source_lines = source.splitlines(keepends=True)

        self.ast_rows: list[AstRow] = []
        self.cst_rows: list[CstRow] = []
        self.metrics = FileMetrics(rel_path=rel_path)

        self._scope_stack: list[str] = []
        self._parent_kinds: list[str] = []
        self._depth: int = 0

    # Generic hooks to track depth and metrics.node_count
    def visit(self, node: cst.CSTNode) -> bool:
        """Track traversal depth, record CST metadata, and continue recursion."""
        self.metrics.node_count += 1
        self._depth += 1
        self.metrics.depths.append(self._depth)
        self._parent_kinds.append(type(node).__name__)
        self._record_cst_row(node)
        return True

    def leave(self, node: cst.CSTNode) -> None:
        """Pop traversal depth and parent stacks after visiting a node."""
        self._parent_kinds.pop()
        self._depth -= 1

    # Complexity heuristics: count control-flow-ish nodes
    def visit_If(self, node: cst.If) -> bool:
        """Increment heuristic complexity for `if` statements."""
        self.metrics.complexity += 1
        return True

    def visit_For(self, node: cst.For) -> bool:
        """Increment heuristic complexity for `for` loops."""
        self.metrics.complexity += 1
        return True

    def visit_While(self, node: cst.While) -> bool:
        """Increment heuristic complexity for `while` loops."""
        self.metrics.complexity += 1
        return True

    def visit_Try(self, node: cst.Try) -> bool:
        """Increment heuristic complexity for `try` statements."""
        self.metrics.complexity += 1
        return True

    def visit_With(self, node: cst.With) -> bool:
        """Increment heuristic complexity for context manager blocks."""
        self.metrics.complexity += 1
        return True

    # AST rows for modules, classes, functions

    def visit_Module(self, node: cst.Module) -> bool:
        """Record the module root and reset traversal scope."""
        qualname = self.module_name
        self._scope_stack = []  # root scope
        self._record_ast_row(
            node=node,
            node_type="Module",
            name=self.module_name.split(".")[-1],
            qualname=qualname,
            parent_qualname=None,
            decorators=[],
        )
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """Record class definitions and push them onto the scope stack."""
        name = node.name.value
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{name}" if parent_qual else f"{self.module_name}.{name}"
        self._scope_stack.append(name)
        self.metrics.class_count += 1
        self._record_ast_row(
            node=node,
            node_type="ClassDef",
            name=name,
            qualname=qualname,
            parent_qualname=parent_qual or self.module_name,
            decorators=[self._decorator_to_str(d.decorator) for d in node.decorators],
        )
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        """Pop the class scope when leaving a class definition."""
        if self._scope_stack:
            self._scope_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """Handle synchronous function definitions."""
        return self._visit_function_like(node, async_kind=False)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Pop scope when exiting a synchronous function."""
        if self._scope_stack:
            self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:
        """Handle asynchronous function definitions."""
        return self._visit_function_like(node, async_kind=True)

    def leave_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
        """Pop scope when exiting an async function."""
        if self._scope_stack:
            self._scope_stack.pop()

    # Helpers

    def _visit_function_like(self, node: cst.BaseStatement, async_kind: bool) -> bool:
        """Shared logic for recording function or async function AST rows."""
        # node.name works for FunctionDef/AsyncFunctionDef
        name = getattr(node, "name", None)
        fn_name = name.value if isinstance(name, cst.Name) else None
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{fn_name}" if parent_qual else f"{self.module_name}.{fn_name}"

        self._scope_stack.append(fn_name or "<lambda>")
        self.metrics.function_count += 1

        decorators = getattr(node, "decorators", []) or []
        self._record_ast_row(
            node=node,
            node_type="AsyncFunctionDef" if async_kind else "FunctionDef",
            name=fn_name,
            qualname=qualname,
            parent_qualname=parent_qual or self.module_name,
            decorators=[self._decorator_to_str(d.decorator) for d in decorators],
        )
        return True

    def _current_qualname(self) -> str | None:
        """Return the current fully qualified name for the traversal scope."""
        if not self._scope_stack:
            return self.module_name
        return f"{self.module_name}." + ".".join(self._scope_stack)

    def _record_ast_row(
        self,
        node: cst.CSTNode,
        *,
        node_type: str,
        name: str | None,
        qualname: str | None,
        parent_qualname: str | None,
        decorators: list[str],
    ) -> None:
        """Append an AstRow describing the current node and its position."""
        pos = self.get_metadata(metadata.PositionProvider, node, None)
        if pos is None:
            lineno = end_lineno = col = end_col = None
        else:
            lineno = pos.start.line
            col = pos.start.column
            end_lineno = pos.end.line
            end_col = pos.end.column

        # Simple stable hash based on location + type + qualname
        h = hashlib.sha1(
            f"{self.rel_path}:{node_type}:{qualname}:{lineno}:{end_lineno}".encode()
        ).hexdigest()

        self.ast_rows.append(
            AstRow(
                path=self.rel_path,
                node_type=node_type,
                name=name,
                qualname=qualname,
                lineno=lineno,
                end_lineno=end_lineno,
                col_offset=col,
                end_col_offset=end_col,
                parent_qualname=parent_qualname,
                decorators=decorators,
                docstring=None,  # can be filled later if needed
                hash=h,
            )
        )

    def _record_cst_row(self, node: cst.CSTNode) -> None:
        """Append a CstRow capturing span, parents, and preview text."""
        pos = self.get_metadata(metadata.PositionProvider, node, None)
        if pos is None:
            return

        start = pos.start
        end = pos.end
        span = {"start": [start.line, start.column], "end": [end.line, end.column]}

        # Derive text preview from span
        try:
            if start.line == end.line:
                line = self.source_lines[start.line - 1]
                snippet = line[start.column : end.column]
            else:
                lines = self.source_lines[start.line - 1 : end.line]
                lines[0] = lines[0][start.column :]
                lines[-1] = lines[-1][: end.column]
                snippet = "".join(lines)
        except Exception:
            snippet = ""

        kind = type(node).__name__
        parents = [k for k in self._parent_kinds[:-1]]  # exclude self
        qnames = [self._current_qualname()] if self._scope_stack else [self.module_name]

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

    @staticmethod
    def _decorator_to_str(expr: cst.CSTNode) -> str:
        """Render a decorator expression into a dotted string representation."""
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            base = AstCstVisitor._decorator_to_str(expr.value)
            return f"{base}.{expr.attr.value}"
        # Fallback: class name of expression
        return type(expr).__name__


def _load_module_map(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> dict[str, str]:
    """Return a mapping of relative file paths to module names from core.modules."""
    rows = con.execute(
        """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    return {path: module for (path, module) in rows}


def ingest_ast_and_cst(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
) -> None:
    """
    Parse modules listed in core.modules and populate AST/CST tables.

    For each module in `core.modules` for the given repo/commit, the function
    reads the source file with LibCST and emits:

      - core.ast_nodes
      - core.ast_metrics
      - core.cst_nodes
    """
    repo_root = repo_root.resolve()
    module_map = _load_module_map(con, repo, commit)
    if not module_map:
        log.warning("No modules found in core.modules for %s@%s", repo, commit)
        return

    # Clear previous AST/CST for safety (single repo per DB assumption).
    con.execute("DELETE FROM core.ast_nodes")
    con.execute("DELETE FROM core.ast_metrics")
    con.execute("DELETE FROM core.cst_nodes")

    insert_ast = """
        INSERT INTO core.ast_nodes (
            path, node_type, name, qualname,
            lineno, end_lineno, col_offset, end_col_offset,
            parent_qualname, decorators, docstring, hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    insert_cst = """
        INSERT INTO core.cst_nodes (
            path, node_id, kind, span, text_preview, parents, qnames
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    insert_metrics = """
        INSERT INTO core.ast_metrics (
            rel_path, node_count, function_count, class_count,
            avg_depth, max_depth, complexity, generated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    now = datetime.now(UTC)

    for rel_path, module_name in module_map.items():
        file_path = repo_root / rel_path
        if not file_path.is_file():
            log.warning("Module path missing on disk: %s", file_path)
            continue

        try:
            source = file_path.read_text(encoding="utf8")
        except UnicodeDecodeError:
            log.warning("Failed to read %s as UTF-8; skipping", file_path)
            continue

        try:
            wrapper = metadata.MetadataWrapper(cst.parse_module(source))
            visitor = AstCstVisitor(rel_path=rel_path, module_name=module_name, source=source)
            wrapper.visit(visitor)
        except Exception as exc:
            log.exception("Failed to parse %s: %s", file_path, exc)
            continue

        # Insert AST rows
        for row in visitor.ast_rows:
            con.execute(
                insert_ast,
                [
                    row.path,
                    row.node_type,
                    row.name,
                    row.qualname,
                    row.lineno,
                    row.end_lineno,
                    row.col_offset,
                    row.end_col_offset,
                    row.parent_qualname,
                    row.decorators,
                    row.docstring,
                    row.hash,
                ],
            )

        # Insert CST rows
        for row in visitor.cst_rows:
            con.execute(
                insert_cst,
                [
                    row.path,
                    row.node_id,
                    row.kind,
                    row.span,
                    row.text_preview,
                    row.parents,
                    row.qnames,
                ],
            )

        # Insert AST metrics
        m = visitor.metrics
        con.execute(
            insert_metrics,
            [
                m.rel_path,
                m.node_count,
                m.function_count,
                m.class_count,
                m.avg_depth,
                m.max_depth,
                m.complexity,
                now,
            ],
        )

    log.info("AST/CST extraction complete for %s@%s", repo, commit)
