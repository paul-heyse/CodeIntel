"""Extract AST/CST structures and metrics into DuckDB tables."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import libcst as cst
from libcst import metadata

log = logging.getLogger(__name__)
FUNCTION_NODE_TYPES = (cst.FunctionDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef))


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
class AstDescriptor:
    """Descriptor capturing logical AST identity for a node."""

    node_type: str
    name: str | None
    qualname: str | None
    parent_qualname: str | None
    decorators: list[str]
    docstring: str | None


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
    span: dict[str, list[int]]
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
    def on_visit(self, node: cst.CSTNode) -> bool:
        """
        Track traversal depth, record CST metadata, and handle AST bookkeeping.

        Returns
        -------
        bool
            True to continue traversal into children.
        """
        self.metrics.node_count += 1
        self._depth += 1
        self.metrics.depths.append(self._depth)
        self._parent_kinds.append(type(node).__name__)
        self._record_cst_row(node)

        if isinstance(node, (cst.If, cst.For, cst.While, cst.Try, cst.With)):
            self.metrics.complexity += 1
        elif isinstance(node, cst.Module):
            self._record_module(node)
        elif isinstance(node, cst.ClassDef):
            self._record_class(node)
        elif isinstance(node, FUNCTION_NODE_TYPES):
            self._record_function(node)

        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Pop traversal depth and parent stacks after visiting a node."""
        if isinstance(original_node, (cst.ClassDef, *FUNCTION_NODE_TYPES)) and self._scope_stack:
            self._scope_stack.pop()
        self._parent_kinds.pop()
        self._depth -= 1

    # Helpers

    def _record_module(self, node: cst.Module) -> None:
        """Record the module root and reset traversal scope."""
        qualname = self.module_name
        self._scope_stack = []  # root scope
        self._record_ast_row(
            node=node,
            descriptor=AstDescriptor(
                node_type="Module",
                name=self.module_name.split(".")[-1],
                qualname=qualname,
                parent_qualname=None,
                decorators=[],
                docstring=self._extract_docstring_from_module(node),
            ),
        )

    def _record_class(self, node: cst.ClassDef) -> None:
        """Record class definitions and push them onto the scope stack."""
        name = node.name.value
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{name}" if parent_qual else f"{self.module_name}.{name}"
        self._scope_stack.append(name)
        self.metrics.class_count += 1
        self._record_ast_row(
            node=node,
            descriptor=AstDescriptor(
                node_type="ClassDef",
                name=name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
                decorators=[self._decorator_to_str(d.decorator) for d in node.decorators],
                docstring=self._extract_docstring_from_suite(node.body),
            ),
        )

    def _record_function(self, node: cst.CSTNode) -> None:
        """Record function or async function AST rows and push scope."""
        if not isinstance(node, FUNCTION_NODE_TYPES):
            return
        name = getattr(node, "name", None)
        fn_name = name.value if isinstance(name, cst.Name) else None
        parent_qual = self._current_qualname()
        qualname = f"{parent_qual}.{fn_name}" if parent_qual else f"{self.module_name}.{fn_name}"

        self._scope_stack.append(fn_name or "<lambda>")
        self.metrics.function_count += 1

        decorators = getattr(node, "decorators", []) or []
        node_type = (
            "AsyncFunctionDef" if isinstance(node, FUNCTION_NODE_TYPES[1]) else "FunctionDef"
        )
        self._record_ast_row(
            node=node,
            descriptor=AstDescriptor(
                node_type=node_type,
                name=fn_name,
                qualname=qualname,
                parent_qualname=parent_qual or self.module_name,
                decorators=[self._decorator_to_str(d.decorator) for d in decorators],
                docstring=self._extract_docstring_from_suite(getattr(node, "body", None)),
            ),
        )

    def _current_qualname(self) -> str:
        """
        Return the current fully qualified name for the traversal scope.

        Returns
        -------
        str
            Fully qualified name including nested scopes.
        """
        if not self._scope_stack:
            return self.module_name
        return f"{self.module_name}." + ".".join(self._scope_stack)

    def _record_ast_row(self, node: cst.CSTNode, *, descriptor: AstDescriptor) -> None:
        """Append an AstRow describing the current node and its position."""
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            lineno = end_lineno = col = end_col = None
        else:
            if not isinstance(pos, metadata.CodeRange):
                lineno = end_lineno = col = end_col = None
            else:
                lineno = pos.start.line
                col = pos.start.column
                end_lineno = pos.end.line
                end_col = pos.end.column

        # Simple stable hash based on location + type + qualname
        h = hashlib.blake2b(
            f"{self.rel_path}:{descriptor.node_type}:{descriptor.qualname}:{lineno}:{end_lineno}".encode(),
            digest_size=16,
        ).hexdigest()

        self.ast_rows.append(
            AstRow(
                path=self.rel_path,
                node_type=descriptor.node_type,
                name=descriptor.name,
                qualname=descriptor.qualname,
                lineno=lineno,
                end_lineno=end_lineno,
                col_offset=col,
                end_col_offset=end_col,
                parent_qualname=descriptor.parent_qualname,
                decorators=descriptor.decorators,
                docstring=descriptor.docstring,
                hash=h,
            )
        )

    def _record_cst_row(self, node: cst.CSTNode) -> None:
        """Append a CstRow capturing span, parents, and preview text."""
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return
        if not isinstance(pos, metadata.CodeRange):
            return

        start = pos.start
        end = pos.end
        span = {"start": [start.line, start.column], "end": [end.line, end.column]}

        # Derive text preview from span with bounds checks
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
        parents = list(self._parent_kinds[:-1])  # exclude self
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
        """
        Render a decorator expression into a dotted string representation.

        Returns
        -------
        str
            Decorator reference rendered as a dotted path.
        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            base = AstCstVisitor._decorator_to_str(expr.value)
            return f"{base}.{expr.attr.value}"
        # Fallback: class name of expression
        return type(expr).__name__

    @staticmethod
    def _extract_docstring_from_module(node: cst.Module) -> str | None:
        """
        Return the module-level docstring, if present.

        Returns
        -------
        str | None
            Evaluated docstring text or None when absent.
        """
        return AstCstVisitor._extract_docstring_from_body(node.body)

    @staticmethod
    def _extract_docstring_from_suite(suite: cst.BaseSuite | None) -> str | None:
        """
        Extract a docstring from an indented or simple suite.

        Returns
        -------
        str | None
            Evaluated docstring text or None when absent.
        """
        if suite is None:
            return None
        if isinstance(suite, cst.IndentedBlock):
            return AstCstVisitor._extract_docstring_from_body(suite.body)
        if isinstance(suite, cst.SimpleStatementSuite):
            return AstCstVisitor._extract_docstring_from_body(suite.body)
        return None

    @staticmethod
    def _extract_docstring_from_body(body: Sequence[cst.CSTNode]) -> str | None:
        """
        Mirror ast.get_docstring semantics for the first statement in a body.

        Returns
        -------
        str | None
            Evaluated docstring text or None when absent.
        """
        if not body:
            return None
        first = body[0]
        if not (isinstance(first, cst.SimpleStatementLine) and first.body):
            return None
        expr = first.body[0]
        if not isinstance(expr, cst.Expr):
            return None
        value = expr.value
        if not isinstance(value, cst.SimpleString):
            return None
        try:
            raw_val = value.evaluated_value
        except ValueError:
            raw_val = None

        if isinstance(raw_val, bytes):
            raw_val = raw_val.decode("utf-8", "replace")
        return raw_val if isinstance(raw_val, str) else None


def _load_module_map(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> dict[str, str]:
    """
    Return a mapping of relative file paths to module names from core.modules.

    Returns
    -------
    dict[str, str]
        Mapping of relative paths to module import names for the repo snapshot.
    """
    rows = con.execute(
        """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    return dict(rows)


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

    # Clear previous AST/CST for this repo/commit only.
    con.execute(
        """
        DELETE FROM core.ast_nodes
        WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)
        """,
        [repo, commit],
    )
    con.execute(
        """
        DELETE FROM core.ast_metrics
        WHERE rel_path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)
        """,
        [repo, commit],
    )
    con.execute(
        """
        DELETE FROM core.cst_nodes
        WHERE path IN (SELECT path FROM core.modules WHERE repo = ? AND commit = ?)
        """,
        [repo, commit],
    )

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
        except Exception:
            log.exception("Failed to parse %s", file_path)
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
